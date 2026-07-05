"""Gate 2.3b — transactional rollback for ``Scheduler._prepare_batch``.

Every scenario in this file exercises the transaction boundary that Gate
2.3b introduces around ``CacheManager.allocate_paged``: if any step
between ``allocate_paged`` and the ``ForwardInput`` return raises, then
the newly-allocated pages and the page_table slice writes performed by
this call are rolled back — while pre-existing prefix / retained pages,
cache handles, ``table_manager`` slots, and running/pending sets stay
exactly as they were.

Tests are hermetic: everything runs on CPU with the naive prefix cache.
Real ``CacheManager`` + real ``allocate_paged`` + real ``_prepare_batch``
are exercised. The only fakes are: engine.graph_runner.pad_batch (just
copies ``batch.reqs`` into ``batch.padded_reqs``), engine.attn_backend
(controls where the raise happens), engine.sampler (returns a sentinel).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"


def _ensure_python_root_on_path() -> None:
    p = str(_PYTHON_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _purge_minisgl_from_sys_modules() -> None:
    for name in list(sys.modules):
        if name == "minisgl" or name.startswith("minisgl."):
            del sys.modules[name]


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def cm_env():
    """Build a real ``CacheManager`` + minimal collaborators on CPU."""
    _ensure_python_root_on_path()
    _purge_minisgl_from_sys_modules()

    import torch

    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.table import TableManager

    page_size = 4
    num_pages = 8  # 8 * 4 = 32 token slots
    max_running = 4
    # page_table shape: [max_running, num_pages * page_size]; entries record the
    # physical slot each logical token position maps to. Init to zeros so we
    # can distinguish "still zero" from "written by allocate_paged".
    page_table = torch.zeros((max_running, num_pages * page_size), dtype=torch.int32)

    tm = TableManager(max_running, page_table)
    # Naive prefix cache always reports cached_len=0 — perfect for tests that
    # only need to exercise the allocation + rollback contract without
    # entangling radix bookkeeping.
    cm = CacheManager(num_pages, page_size, page_table, type="naive")

    return {
        "torch": torch,
        "cm": cm,
        "tm": tm,
        "page_table": page_table,
        "page_size": page_size,
        "num_pages": num_pages,
    }


def _make_req(env, *, table_idx: int, cached_len: int, device_len: int, uid: int):
    """Build a real ``Req`` that ``allocate_paged`` will happily consume."""
    from minisgl.core import Req, SamplingParams
    from minisgl.kvcache.naive_cache import NaiveCacheHandle

    # NaiveCacheHandle carries the empty tensor plumbed in by NaivePrefixCache
    # on construction; ``env['cm']`` already did that plumbing.
    assert 0 <= cached_len < device_len
    input_ids = env["torch"].zeros(device_len, dtype=env["torch"].int32)
    output_len = 1  # any positive value satisfies Req's assert
    return Req(
        input_ids=input_ids,
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        cache_handle=NaiveCacheHandle(),
    )


def _snapshot(env) -> dict:
    """Snapshot every allocator/page_table quantity we assert on."""
    return {
        "free_pages": int(len(env["cm"].free_slots)),
        "available_size": int(env["cm"].available_size),
        "free_slots_sorted": sorted(int(x) for x in env["cm"].free_slots.tolist()),
        "table_free": sorted(env["tm"]._free_slots),
        "page_table": env["page_table"].detach().cpu().clone(),
    }


def _assert_state_restored(before: dict, after: dict) -> None:
    assert after["free_pages"] == before["free_pages"], (
        f"free_pages leaked: {before['free_pages']} -> {after['free_pages']}"
    )
    assert after["available_size"] == before["available_size"], (
        f"available_size drift: {before['available_size']} -> {after['available_size']}"
    )
    assert after["free_slots_sorted"] == before["free_slots_sorted"], (
        f"free_slots contents drifted:\n  before={before['free_slots_sorted']}"
        f"\n  after ={after['free_slots_sorted']}"
    )
    assert after["table_free"] == before["table_free"], (
        f"table_manager free list drifted: {before['table_free']} vs {after['table_free']}"
    )
    import torch as _t

    assert _t.equal(before["page_table"], after["page_table"]), (
        "page_table not fully restored after rollback"
    )


# ---------------------------------------------------------- scheduler harness


def _build_scheduler_skeleton(env, *, prepare_metadata: Callable[[object], None]):
    """Instantiate ``Scheduler`` via ``__new__`` and bolt on just enough for
    ``_prepare_batch`` to run end-to-end.
    """
    _ensure_python_root_on_path()

    from minisgl.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.cache_manager = env["cm"]
    sched.table_manager = env["tm"]
    sched.device = env["page_table"].device

    engine = MagicMock()
    engine.page_table = env["page_table"]

    # graph_runner.pad_batch: copy reqs into padded_reqs (no cuda-graph padding).
    def _pad_batch(batch):
        batch.padded_reqs = list(batch.reqs)

    engine.graph_runner.pad_batch = _pad_batch

    engine.attn_backend.prepare_metadata = prepare_metadata

    # Return a lightweight sentinel; _prepare_batch stores whatever it gets.
    engine.sampler.prepare = lambda batch: ("sample_args_sentinel", batch.size)

    sched.engine = engine
    return sched


def _new_prefill_batch(reqs):
    from minisgl.core import Batch

    return Batch(reqs=list(reqs), phase="prefill")


# =========================================================================
# A. No prefix — prepare_metadata raises after allocation succeeds
# =========================================================================


def test_A_no_prefix_rollback_restores_pages_and_page_table(cm_env):
    """B=2 with cached_len=0, prepare_metadata raises → pages + slice both
    restored, table_manager untouched, batch scratch cleared."""
    env = cm_env
    # Reserve two table rows via the real TableManager (mirror scheduler use).
    ta = env["tm"].allocate()
    tb = env["tm"].allocate()
    reqs = [
        _make_req(env, table_idx=ta, cached_len=0, device_len=5, uid=1),   # 2 pages
        _make_req(env, table_idx=tb, cached_len=0, device_len=3, uid=2),   # 1 page
    ]
    before = _snapshot(env)

    def _explode(batch):
        raise RuntimeError("gate23b_A: injected failure inside prepare_metadata")

    sched = _build_scheduler_skeleton(env, prepare_metadata=_explode)
    batch = _new_prefill_batch(reqs)

    with pytest.raises(RuntimeError, match="gate23b_A"):
        sched._prepare_batch(batch)

    after = _snapshot(env)
    _assert_state_restored(before, after)

    # Batch's scratch attrs must be scrubbed so a later scheduling round
    # cannot accidentally reuse a stale positions / out_loc / metadata.
    for attr in ("positions", "out_loc", "padded_reqs", "attn_metadata"):
        assert attr not in batch.__dict__, f"batch.{attr} not cleared after rollback"

    # table_manager slots must NOT have been freed — the reqs still own them.
    assert ta not in env["tm"]._free_slots
    assert tb not in env["tm"]._free_slots


# =========================================================================
# B. With retained prefix — only the new page is rolled back
# =========================================================================


def test_B_with_prefix_rollback_preserves_prefix_pages(cm_env):
    """Simulate a request that already owns one retained/cached page. After
    a failed allocation, only the newly-allocated page is returned; the
    prefix page slot in page_table and free_slots is untouched."""
    env = cm_env
    ta = env["tm"].allocate()

    # Simulate a prior prefix: consume physical page 3 by hand and write it
    # into page_table[ta, 0:page_size]. This mirrors what a previous
    # allocate_paged call would have left behind.
    import torch as _t

    page_size = env["page_size"]
    # Pop page 3 from free_slots and pretend it's retained by ta.
    all_free_before_manual = env["cm"].free_slots.tolist()
    assert 3 * page_size in all_free_before_manual
    # Remove that entry, keep the rest.
    keep = [x for x in all_free_before_manual if x != 3 * page_size]
    env["cm"].free_slots = _t.tensor(keep, dtype=_t.int32)
    env["page_table"][ta, 0:page_size] = _t.arange(
        3 * page_size, 3 * page_size + page_size, dtype=_t.int32
    )

    # Snapshot the "retained prefix" state — this is what MUST survive.
    prefix_before = env["page_table"][ta, 0:page_size].clone()
    before = _snapshot(env)

    # Now build a req with cached_len=page_size (== retained prefix length)
    # and device_len=page_size+1 → needs exactly one new page starting at
    # logical position page_size.
    reqs = [
        _make_req(
            env, table_idx=ta,
            cached_len=page_size, device_len=page_size + 1, uid=10,
        ),
    ]

    def _explode(batch):
        raise NotImplementedError("gate23b_B: mimicking Ascend FIA refusal")

    sched = _build_scheduler_skeleton(env, prepare_metadata=_explode)
    batch = _new_prefill_batch(reqs)

    with pytest.raises(NotImplementedError, match="gate23b_B"):
        sched._prepare_batch(batch)

    after = _snapshot(env)
    _assert_state_restored(before, after)

    # Prefix page contents preserved bit-equal.
    prefix_after = env["page_table"][ta, 0:page_size]
    assert _t.equal(prefix_before, prefix_after), (
        f"prefix page mutated: {prefix_before} -> {prefix_after}"
    )
    # Retained physical page 3 must NOT be back on the free list.
    assert 3 * page_size not in env["cm"].free_slots.tolist()


# =========================================================================
# C. Two requests with different new-page counts — all-or-nothing rollback
# =========================================================================


def test_C_two_reqs_different_page_counts_all_rolled_back(cm_env):
    """Two reqs allocate 3 pages and 1 page respectively; failure returns
    ALL 4 pages, restores BOTH page_table slices, and does so without
    double-freeing any page."""
    env = cm_env
    ta = env["tm"].allocate()
    tb = env["tm"].allocate()
    page_size = env["page_size"]

    reqs = [
        # 3 new pages (device_len 9 spans logical pages 0..2 inclusive)
        _make_req(env, table_idx=ta, cached_len=0, device_len=9, uid=100),
        # 1 new page (device_len 2 spans logical page 0 only)
        _make_req(env, table_idx=tb, cached_len=0, device_len=2, uid=101),
    ]
    before = _snapshot(env)

    def _explode(batch):
        raise RuntimeError("gate23b_C: injected failure")

    sched = _build_scheduler_skeleton(env, prepare_metadata=_explode)
    batch = _new_prefill_batch(reqs)

    with pytest.raises(RuntimeError, match="gate23b_C"):
        sched._prepare_batch(batch)

    after = _snapshot(env)
    _assert_state_restored(before, after)

    # Duplicate-free check: sorted free_slots must have no repeated entries.
    slots = after["free_slots_sorted"]
    assert len(slots) == len(set(slots)), (
        f"double free detected in free_slots: {slots}"
    )
    # And every entry must still be page-aligned.
    assert all(x % page_size == 0 for x in slots)


# =========================================================================
# D. Success path — behaviour identical to pre-Gate-2.3b
# =========================================================================


def test_D_success_path_pages_retained_and_forward_input_populated(cm_env):
    """Metadata succeeds → new pages stay allocated, page_table stays
    written, no rollback runs, ForwardInput carries the sampler sentinel."""
    env = cm_env
    ta = env["tm"].allocate()
    tb = env["tm"].allocate()
    reqs = [
        _make_req(env, table_idx=ta, cached_len=0, device_len=5, uid=200),  # 2 pages
        _make_req(env, table_idx=tb, cached_len=0, device_len=3, uid=201),  # 1 page
    ]
    before = _snapshot(env)

    def _ok(batch):
        # Set an attn_metadata sentinel so downstream can observe success.
        batch.attn_metadata = "meta_sentinel"

    sched = _build_scheduler_skeleton(env, prepare_metadata=_ok)
    batch = _new_prefill_batch(reqs)

    fi = sched._prepare_batch(batch)

    # 3 pages consumed total (2 + 1).
    after = _snapshot(env)
    assert after["free_pages"] == before["free_pages"] - 3, (
        f"expected 3 pages consumed, got "
        f"{before['free_pages']} -> {after['free_pages']}"
    )
    # Sampler sentinel propagated.
    assert fi.sample_args == ("sample_args_sentinel", 2)
    # Batch retains its scratch attrs on the success path.
    assert hasattr(batch, "positions")
    assert hasattr(batch, "out_loc")
    assert hasattr(batch, "padded_reqs")
    assert batch.attn_metadata == "meta_sentinel"
    # page_table row for ta got written into positions [0..5) — non-trivial.
    assert env["page_table"][ta, :5].abs().sum().item() > 0
    assert env["page_table"][tb, :3].abs().sum().item() > 0


# =========================================================================
# E. Second call after a rollback still succeeds
# =========================================================================


def test_E_rollback_then_success_leaves_allocator_healthy(cm_env):
    """Fail once, roll back, then re-run _prepare_batch with a success
    metadata callback. Must succeed and produce exactly the same effect as
    if the failed call had never happened."""
    env = cm_env
    ta = env["tm"].allocate()
    tb = env["tm"].allocate()

    baseline = _snapshot(env)

    # ---- Round 1: injected failure ----
    reqs_fail = [
        _make_req(env, table_idx=ta, cached_len=0, device_len=5, uid=300),
        _make_req(env, table_idx=tb, cached_len=0, device_len=3, uid=301),
    ]

    def _explode(batch):
        raise RuntimeError("gate23b_E: round1 boom")

    sched1 = _build_scheduler_skeleton(env, prepare_metadata=_explode)
    batch1 = _new_prefill_batch(reqs_fail)
    with pytest.raises(RuntimeError, match="gate23b_E"):
        sched1._prepare_batch(batch1)

    after_round1 = _snapshot(env)
    _assert_state_restored(baseline, after_round1)

    # ---- Round 2: same reqs, this time metadata succeeds ----
    reqs_ok = [
        _make_req(env, table_idx=ta, cached_len=0, device_len=5, uid=302),
        _make_req(env, table_idx=tb, cached_len=0, device_len=3, uid=303),
    ]

    def _ok(batch):
        batch.attn_metadata = "round2_ok"

    sched2 = _build_scheduler_skeleton(env, prepare_metadata=_ok)
    batch2 = _new_prefill_batch(reqs_ok)
    fi = sched2._prepare_batch(batch2)

    after_round2 = _snapshot(env)
    # 3 pages consumed after round 2 — matches what round 1 would have done
    # if it had succeeded.
    assert after_round2["free_pages"] == baseline["free_pages"] - 3, (
        f"post-round2 page count wrong: {baseline['free_pages']}"
        f" -> {after_round2['free_pages']}"
    )
    assert fi.batch is batch2
    assert batch2.attn_metadata == "round2_ok"
