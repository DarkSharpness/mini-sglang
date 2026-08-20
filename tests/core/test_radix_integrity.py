"""RadixPrefixCache.check_integrity walks the tree (#149)."""

from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    page_table = torch.empty((1,))
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _allocate_and_insert(cm: CacheManager, n_tokens: int, id_offset: int = 0):
    """Take free pages, then file them in radix (keeps free_slots and tree disjoint)."""
    n_pages = n_tokens // cm.page_size
    pages = cm._allocate(n_pages)
    indices = cm._page_to_token(pages)
    input_ids = torch.arange(id_offset, id_offset + n_tokens, dtype=torch.int32)
    result = cm.prefix_cache.insert_prefix(input_ids, indices)
    return input_ids, indices, result


def test_empty_tree_ok():
    _make_cache_manager(4, 4).check_integrity()


def test_insert_then_ok():
    cm = _make_cache_manager(8, 4)
    _allocate_and_insert(cm, 8)
    cm.check_integrity()


def test_shared_prefix_unique_pages():
    """Warrior vs wizard: shared opening, distinct tails."""
    cm = _make_cache_manager(8, 4)
    head_ids, head_idx, _ = _allocate_and_insert(cm, 8)

    extra_pages = cm._allocate(1)
    extra_idx = cm._page_to_token(extra_pages)
    tail_ids = torch.cat([head_ids, torch.arange(8, 12, dtype=torch.int32)])
    tail_idx = torch.cat([head_idx, extra_idx])
    cm.prefix_cache.insert_prefix(tail_ids, tail_idx)
    cm.check_integrity()


def test_evict_tail_then_ok():
    cm = _make_cache_manager(8, 4)
    head_ids, head_idx, _ = _allocate_and_insert(cm, 8)
    extra_pages = cm._allocate(1)
    extra_idx = cm._page_to_token(extra_pages)
    cm.prefix_cache.insert_prefix(
        torch.cat([head_ids, torch.arange(8, 12, dtype=torch.int32)]),
        torch.cat([head_idx, extra_idx]),
    )
    evicted = cm.prefix_cache.evict(4)
    cm.free_slots = torch.cat([cm.free_slots, evicted[:: cm.page_size]])
    cm.check_integrity()


def test_lock_splits_protected_vs_evictable():
    cm = _make_cache_manager(8, 4)
    _, _, result = _allocate_and_insert(cm, 8)
    cm.prefix_cache.lock_handle(result.handle)
    assert cm.prefix_cache.size_info.protected_size == 8
    assert cm.prefix_cache.size_info.evictable_size == 0
    cm.check_integrity()
    cm.prefix_cache.lock_handle(result.handle, unlock=True)
    cm.check_integrity()


def test_tree_page_also_in_free_slots_fails():
    cm = _make_cache_manager(8, 4)
    _allocate_and_insert(cm, 8)
    cm.check_integrity()
    tree_starts = cm.prefix_cache.cached_token_indices()[:: cm.page_size]
    # Keep free_slots length so the count check still passes; only overlap.
    cm.free_slots = cm.free_slots.clone()
    cm.free_slots[0] = tree_starts[0]
    with pytest.raises(RuntimeError, match="tree and free_slots"):
        cm.check_integrity()


def test_duplicate_index_fails():
    cm = _make_cache_manager(8, 4)
    head_ids, head_idx, _ = _allocate_and_insert(cm, 8)
    extra_pages = cm._allocate(1)
    extra_idx = cm._page_to_token(extra_pages)
    cm.prefix_cache.insert_prefix(
        torch.cat([head_ids, torch.arange(8, 12, dtype=torch.int32)]),
        torch.cat([head_idx, extra_idx]),
    )
    nodes = cm.prefix_cache._iter_non_root_nodes()
    assert len(nodes) >= 2
    nodes[1]._value[0] = int(nodes[0].value[0].item())
    with pytest.raises(RuntimeError, match="duplicate"):
        cm.prefix_cache.check_integrity()
