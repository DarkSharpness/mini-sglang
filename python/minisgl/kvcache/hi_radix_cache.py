from __future__ import annotations

import heapq
import time
from typing import List

import torch

from .base import BaseCacheHandle, MatchResult
from .host_pool import HostKVCachePool
from .radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode


class HiRadixPrefixCache(RadixPrefixCache):
    def __init__(self, device: torch.device, host_pool: HostKVCachePool | None = None):
        super().__init__(device)
        self.host_pool = host_pool
        self.evictable_host_size = 0
        self.transfer_stream = torch.cuda.Stream(device=device)

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)

        if not node.evicted:
            return MatchResult(RadixCacheHandle(prefix_len, node))

        host_handle = self._build_host_handle(node)

        while not node.is_root() and node.evicted:
            node = node.parent

        device_len = self._collect_device_len(node)
        return MatchResult(RadixCacheHandle(device_len, node), host_handle)

    def _collect_device_len(self, node: RadixTreeNode) -> int:
        length = 0
        cur = node
        while not cur.is_root():
            if not cur.evicted:
                length += cur.length
            cur = cur.parent
        return length

    def _build_host_handle(self, node: RadixTreeNode) -> BaseCacheHandle:
        indices_list: List[torch.Tensor] = []
        length_list: List[int] = []
        cur = node
        while not cur.is_root():
            if not cur.evicted:
                break
            if cur.backuped:
                indices_list.append(cur.host_value)
                length_list.append(cur.length)
            cur = cur.parent
        indices_list.reverse()
        length_list.reverse()
        host_len = sum(length_list)
        host_page_indices = torch.cat(indices_list) if indices_list else torch.empty(0, dtype=torch.int32)

        class HostCacheHandle(BaseCacheHandle):
            def __init__(self, cached_len: int, page_indices: torch.Tensor, end_node: RadixTreeNode):
                super().__init__(cached_len=cached_len)
                self.page_indices = page_indices
                self.end_node = end_node

            def get_matched_indices(self) -> torch.Tensor:
                return self.page_indices

        return HostCacheHandle(host_len, host_page_indices, node)

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert size <= self.evictable_size

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leave_nodes
            node = heapq.heappop(leave_nodes)
            if node.evicted:
                continue
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            self.evictable_size -= node.length

            if not node.backuped and self.host_pool is not None:
                self._write_to_host(node)

            if node.backuped:
                evicted_indices.append(node.value)
                node._value = None
            else:
                evicted_indices.append(node.value)
                parent = node.parent
                del parent.children[self.key_fn(node._key)]
                if parent.is_leaf() and parent.ref_count == 0:
                    heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def _write_to_host(self, node: RadixTreeNode) -> None:
        num_pages = len(node.value) // self.page_size
        host_page_indices = self._alloc_host_pages(num_pages)
        device_indices = node.value.clone()
        self.host_pool.copy_from_device(device_indices, host_page_indices, self.transfer_stream)
        self.transfer_stream.synchronize()
        node._host_value = host_page_indices
        self.evictable_host_size += node.length

    def _alloc_host_pages(self, num_pages: int) -> torch.Tensor:
        host_indices = self.host_pool.alloc(num_pages)
        if host_indices is not None:
            return host_indices
        self._evict_host(num_pages * self.page_size)
        host_indices = self.host_pool.alloc(num_pages)
        assert host_indices is not None
        return host_indices

    def evict_host(self, size: int) -> torch.Tensor:
        if size == 0 or self.evictable_host_size == 0:
            return self.empty_tensor

        host_leaves = self._collect_host_leaves()
        heapq.heapify(host_leaves)
        freed_size = 0

        while freed_size < size and host_leaves:
            node = heapq.heappop(host_leaves)
            if not node.evicted or not node.backuped or node.ref_count > 0:
                continue
            freed_size += node.length
            self.host_pool.free(node.host_value)
            self.evictable_host_size -= node.length
            node._host_value = None

            if node.is_leaf():
                parent = node.parent
                del parent.children[self.key_fn(node._key)]
                self._try_merge_parent(parent)

        return self.empty_tensor

    def _evict_host(self, needed_size: int) -> None:
        while self.host_pool.available_pages * self.page_size < needed_size and self.evictable_host_size > 0:
            self.evict_host(needed_size)

    def promote_to_device(self, host_handle: BaseCacheHandle, device_indices: torch.Tensor) -> RadixCacheHandle:
        host_page_indices = host_handle.get_matched_indices()
        host_len = host_handle.cached_len

        self.host_pool.copy_to_device(host_page_indices, device_indices[:host_len], self.transfer_stream)
        self.transfer_stream.synchronize()

        self.host_pool.free(host_page_indices)
        self.evictable_host_size -= host_len

        end_node = host_handle.end_node
        self._restore_node_values(end_node, device_indices[:host_len])

        total_len = self._collect_device_len(end_node)
        return RadixCacheHandle(total_len, end_node)

    def _restore_node_values(self, node: RadixTreeNode, device_indices: torch.Tensor) -> None:
        nodes: List[RadixTreeNode] = []
        cur = node
        while not cur.is_root() and cur.evicted:
            nodes.append(cur)
            cur = cur.parent
        nodes.reverse()

        offset = 0
        for n in nodes:
            length = n.length
            n._value = device_indices[offset : offset + length].clone()
            n._host_value = None
            n.timestamp = time.monotonic_ns()
            if n.ref_count == 0:
                self.evictable_size += length
            else:
                self.protected_size += length
            offset += length

    def _collect_host_leaves(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        host_leaves: List[RadixTreeNode] = []
        while nodes:
            node = nodes.pop()
            if node.evicted and node.backuped and node.is_leaf() and node.ref_count == 0:
                host_leaves.append(node)
            for child in node.children.values():
                nodes.append(child)
        return host_leaves

    def _try_merge_parent(self, parent: RadixTreeNode) -> None:
        if not parent.is_root() and parent.is_leaf() and parent.evicted and not parent.backuped and parent.ref_count == 0:
            grandparent = parent.parent
            del grandparent.children[self.key_fn(parent._key)]
            self._try_merge_parent(grandparent)
