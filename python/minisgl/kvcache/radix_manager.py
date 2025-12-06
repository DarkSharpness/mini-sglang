from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[int(self._key[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # root is always protected
        self.evictable_size = 0
        self.protected_size = 0

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node = node.parent
                node.ref_count -= 1
                assert node.ref_count > 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
        else:
            while not node.is_root():
                node = node.parent
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return RadixCacheHandle(prefix_len, node), torch.cat(value_list)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass
