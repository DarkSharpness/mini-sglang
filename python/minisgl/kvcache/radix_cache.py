from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
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
        parent.children[self.key_fn(self._key)] = self

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
        try:
            from minisgl.kernel import fast_compare_key

            # compare key and input_ids, find the first diff
            return fast_compare_key(self._key, input_ids)
        except (FileNotFoundError, ModuleNotFoundError, OSError, ImportError):
            return _cpu_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
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

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
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
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

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
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def cached_token_indices(self) -> torch.Tensor:
        """All page-table token indices owned by the tree (not GPU K/V)."""
        chunks: List[torch.Tensor] = []
        for node in self._iter_non_root_nodes():
            chunks.append(node.value)
        if not chunks:
            return self.empty_tensor
        return torch.cat(chunks)

    def check_integrity(self) -> None:
        evictable = 0
        protected = 0
        seen: set[int] = set()

        for node in self._iter_non_root_nodes():
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length

            for idx in node.value.tolist():
                if idx in seen:
                    raise RuntimeError(f"Radix integrity: duplicate page-table index {idx}")
                seen.add(idx)

        if evictable != self.evictable_size or protected != self.protected_size:
            raise RuntimeError(
                "Radix integrity: size mismatch "
                f"walk evictable={evictable} protected={protected}, "
                f"stored evictable={self.evictable_size} protected={self.protected_size}"
            )

    def _iter_non_root_nodes(self) -> List[RadixTreeNode]:
        out: List[RadixTreeNode] = []
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                stack.append(child)
            if not node.is_root():
                out.append(node)
        return out

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

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
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
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _cpu_compare_key(key: torch.Tensor, input_ids: torch.Tensor) -> int:
    """Same semantics as radix.cpp fast_compare_key; used when AOT/ninja is unavailable."""
    n = min(len(key), len(input_ids))
    if n == 0:
        return 0
    neq = key[:n] != input_ids[:n]
    hits = torch.nonzero(neq, as_tuple=False)
    if len(hits) == 0:
        return n
    return int(hits[0].item())


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
