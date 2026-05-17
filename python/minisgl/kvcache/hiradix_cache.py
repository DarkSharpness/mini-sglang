from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down, init_logger

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]

logger = init_logger(__name__)


class HiRadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, HiRadixTreeNode] = {}
        self._parent: HiRadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = HiRadixTreeNode.counter
        HiRadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._cuda_value: torch.Tensor | None = None
        self._host_value: torch.Tensor | None = None
        self._length: int

    def on_cuda_only(self) -> bool:
        return self._cuda_value is not None and self._host_value is None

    def on_host_only(self) -> bool:
        return self._cuda_value is None and self._host_value is not None

    def set_key_value(
        self,
        key: torch.Tensor,
        cuda_value: torch.Tensor | None,
        host_value: torch.Tensor | None = None,
    ) -> None:
        self._key = key
        self._cuda_value = cuda_value
        self._host_value = host_value
        self._length = len(key)
        assert self._length > 0, "Node length must be greater than 0"

    def set_parent(self, parent: HiRadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> HiRadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def cuda_value(self) -> torch.Tensor:
        assert self._cuda_value is not None
        return self._cuda_value

    @property
    def host_value(self) -> torch.Tensor:
        assert self._host_value is not None
        return self._host_value

    @cuda_value.setter
    def cuda_value(self, value: torch.Tensor | None) -> None:
        if value is not None:
            assert self._cuda_value is None and len(value) == self.length
        self._cuda_value = value

    @host_value.setter
    def host_value(self, value: torch.Tensor | None) -> None:
        if value is not None:
            assert self._host_value is None and len(value) == self.length
        self._host_value = value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf_device(self) -> bool:
        return all(c._cuda_value is None for c in self.children.values())

    def is_leaf_host(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> HiRadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = HiRadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(
            self._key[:pos],
            _maybe_slice(self._cuda_value, slice(0, pos)),
            _maybe_slice(self._host_value, slice(0, pos)),
        )
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        self.set_key_value(
            self._key[pos:],
            _maybe_slice(self._cuda_value, slice(pos, None)),
            _maybe_slice(self._host_value, slice(pos, None)),
        )
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: HiRadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class HiRadixCacheHandle(BaseCacheHandle):
    node: HiRadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.cuda_value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class HiRadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = HiRadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        assert node.is_root() or not node.on_host_only()
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        host_node, host_prefix_len = self._tree_walk(input_ids)
        cuda_node, cuda_prefix_len = host_node, host_prefix_len
        while not cuda_node.is_root() and cuda_node.on_host_only():
            cuda_prefix_len -= cuda_node.length
            cuda_node = cuda_node.parent
        return MatchResult(
            cuda_handle=HiRadixCacheHandle(cuda_prefix_len, cuda_node),
            host_handle=HiRadixCacheHandle(host_prefix_len, host_node),
        )

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        host_node, host_prefix_len = self._tree_walk(input_ids)
        cuda_node, cuda_prefix_len = host_node, host_prefix_len
        while not cuda_node.is_root() and cuda_node.on_host_only():
            cuda_prefix_len -= cuda_node.length
            cuda_node = cuda_node.parent
        self.evictable_size += host_prefix_len - cuda_prefix_len
        updated_indices = indices[cuda_prefix_len:host_prefix_len].clone()
        node = host_node
        while not node.is_root() and node.on_host_only():
            node.cuda_value = updated_indices[-node.length :]
            updated_indices = updated_indices[: -node.length]
            node = node.parent
        assert len(updated_indices) == 0
        if host_prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = HiRadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[host_prefix_len:], indices[host_prefix_len:].clone())
            new_node.set_parent(host_node)
            self.evictable_size += new_node.length
            host_node = new_node
        return InsertResult(cuda_prefix_len, HiRadixCacheHandle(insert_len, host_node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict(is_host=False)
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert len(leave_nodes) > 0, "Not enough evictable nodes"
            node = heapq.heappop(leave_nodes)
            evicted_size += node.length
            evicted_indices.append(node.cuda_value)
            self.evictable_size -= node.length
            parent = node.parent
            if node.on_cuda_only():  # no backup on host, remove the node
                del parent.children[self.key_fn(node._key)]
            else:  # evict device part, but keep host backup
                node.cuda_value = None
            # NOTE: root is always protected, so won't be evicted
            if parent.ref_count == 0 and parent.is_leaf_device():
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def try_evict_host(self, size: int) -> List[torch.Tensor]:
        if size == 0:
            return []

        leave_nodes = self._collect_leave_nodes_for_evict(is_host=True)
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size and leave_nodes:
            node = heapq.heappop(leave_nodes)
            if not node.on_host_only():  # still has device backup, skip eviction
                continue

            evicted_size += node.length
            evicted_indices.append(node.host_value)
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.ref_count == 0 and parent.is_leaf_host():
                heapq.heappush(leave_nodes, parent)

        return evicted_indices

    def get_writable_length(self, handle: BaseCacheHandle) -> int:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        needed_len = 0
        while not node.is_root() and node.on_cuda_only():
            needed_len += node.length
            node = node.parent
        return needed_len

    def set_host(self, handle: BaseCacheHandle, indices: torch.Tensor) -> List[torch.Tensor]:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        result: List[torch.Tensor] = []
        while not node.is_root() and node.on_cuda_only():
            node.host_value = indices[-node.length :]
            indices = indices[: -node.length]
            result.append(node.cuda_value)
            node = node.parent
        assert len(indices) == 0
        result.reverse()
        return result

    def set_cuda(self, handle: BaseCacheHandle, indices: torch.Tensor) -> List[torch.Tensor]:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        result: List[torch.Tensor] = []
        self.evictable_size += node.length  # update evictable size first
        while not node.is_root() and node.on_host_only():
            assert node.ref_count == 0
            node.cuda_value = indices[-node.length :]
            indices = indices[: -node.length]
            result.append(node.host_value)
            node = node.parent
        assert len(indices) == 0
        result.reverse()
        return result

    def reset(self) -> None:
        raise NotImplementedError("HiRadixPrefixCache.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self, is_host: bool) -> List[HiRadixTreeNode]:
        nodes: List[HiRadixTreeNode] = list(self.root_node.children.values())
        leave_nodes: List[HiRadixTreeNode] = []

        fn = HiRadixTreeNode.is_leaf_host if is_host else HiRadixTreeNode.is_leaf_device
        while len(nodes) > 0:
            node = nodes.pop()
            if not is_host and node._cuda_value is None:
                continue
            if fn(node):
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)
        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[HiRadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


def _maybe_slice(t: torch.Tensor | None, s) -> torch.Tensor | None:
    return t[s] if t is not None else None
