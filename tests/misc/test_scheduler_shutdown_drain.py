"""Gate 2.3d — ``Scheduler._drain_requests`` shutdown drain.

Exercises the drain helper directly on a bare-bones Scheduler skeleton
built via ``__new__``. Uses the real ``CacheManager`` (with the naive
prefix cache), the real ``TableManager``, the real ``PrefillManager``,
the real ``DecodeManager``, and real ``Req`` / ``PendingReq`` / ``ChunkedReq``
objects. Only the outer ``shutdown()`` collaborators (engine, distributed
sync) are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
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
def sched_env():
    _ensure_python_root_on_path()
    _purge_minisgl_from_sys_modules()

    import torch

    from minisgl.core import Req, SamplingParams
    from minisgl.kvcache.naive_cache import NaiveCacheHandle
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.decode import DecodeManager
    from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.table import TableManager
    from minisgl.scheduler.utils import PendingReq

    page_size = 4
    num_pages = 8
    max_running = 4
    page_table = torch.zeros((max_running, num_pages * page_size), dtype=torch.int32)

    tm = TableManager(max_running, page_table)
    cm = CacheManager(num_pages, page_size, page_table, type="naive")
    dm = DecodeManager(page_size)
    pm = PrefillManager(cache_manager=cm, table_manager=tm, decode_manager=dm)

    sched = Scheduler.__new__(Scheduler)
    sched.cache_manager = cm
    sched.table_manager = tm
    sched.decode_manager = dm
    sched.prefill_manager = pm
    sched.finished_reqs = set()

    return {
        "torch": torch,
        "sched": sched,
        "cm": cm,
        "tm": tm,
        "dm": dm,
        "pm": pm,
        "page_table": page_table,
        "page_size": page_size,
        "num_pages": num_pages,
        "Req": Req,
        "ChunkedReq": ChunkedReq,
        "PendingReq": PendingReq,
        "SamplingParams": SamplingParams,
        "NaiveCacheHandle": NaiveCacheHandle,
    }


def _make_req(env, *, cls, cached_len: int, device_len: int, uid: int):
    tm = env["tm"]
    cm = env["cm"]
    # Reserve a table row + pages the way PrefillAdder + allocate_paged would.
    table_idx = tm.allocate()
    # Allocate pages for this req and write them into the page_table.
    reqs = [
        cls(
            input_ids=env["torch"].zeros(device_len, dtype=env["torch"].int32),
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=8,
            uid=uid,
            sampling_params=env["SamplingParams"](temperature=0.0, max_tokens=1),
            cache_handle=env["NaiveCacheHandle"](),
        )
    ]
    cm.allocate_paged(reqs)
    return reqs[0]


def _make_pending(env, *, uid: int, input_len: int, chunked_req=None):
    input_ids = env["torch"].zeros(input_len, dtype=env["torch"].int32)
    pending = env["PendingReq"](
        uid=uid,
        input_ids=input_ids,
        sampling_params=env["SamplingParams"](temperature=0.0, max_tokens=4),
    )
    pending.chunked_req = chunked_req
    return pending


def _snapshot(env) -> dict:
    return {
        "free_pages": int(len(env["cm"].free_slots)),
        "available_size": int(env["cm"].available_size),
        "free_slots_sorted": sorted(int(x) for x in env["cm"].free_slots.tolist()),
        "table_free_sorted": sorted(env["tm"]._free_slots),
        "pending_len": len(env["pm"].pending_list),
        "running_len": len(env["dm"].running_reqs),
    }


# =========================================================================
# A. Only unresourced waiting pending reqs — drop from pending_list.
# =========================================================================


def test_A_waiting_only_drops_pending_list(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    env["pm"].pending_list.append(_make_pending(env, uid=1, input_len=4))
    env["pm"].pending_list.append(_make_pending(env, uid=2, input_len=6))

    assert len(env["pm"].pending_list) == 2
    env["sched"]._drain_requests()

    after = _snapshot(env)
    assert after == baseline, (
        f"drain changed allocator state despite no allocated resources: "
        f"{baseline} -> {after}"
    )


# =========================================================================
# B. Pending with chunked_req — release table + KV via _free_req_resources.
# =========================================================================


def test_B_pending_with_chunked_req_releases_resources(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    chunked = _make_req(
        env, cls=env["ChunkedReq"], cached_len=0, device_len=5, uid=10
    )
    pending = _make_pending(env, uid=10, input_len=8, chunked_req=chunked)
    env["pm"].pending_list.append(pending)

    # Verify pre-drain: one page consumed (5 tokens spanning 2 pages of size 4).
    pre = _snapshot(env)
    assert pre["free_pages"] < baseline["free_pages"]
    assert 0 not in env["tm"]._free_slots or len(env["tm"]._free_slots) < len(
        baseline["table_free_sorted"]
    )

    env["sched"]._drain_requests()

    after = _snapshot(env)
    assert after["pending_len"] == 0
    assert after["running_len"] == 0
    assert after["free_pages"] == baseline["free_pages"], (
        f"KV pages not fully returned: {baseline['free_pages']} != {after['free_pages']}"
    )
    assert after["table_free_sorted"] == baseline["table_free_sorted"], (
        f"table_free drift after drain: {after['table_free_sorted']}"
    )
    env["cm"].check_integrity()


# =========================================================================
# C. Only running decode reqs — clean them out via decode_manager.
# =========================================================================


def test_C_running_only_releases_via_decode_manager(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    r1 = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=100)
    r2 = _make_req(env, cls=env["Req"], cached_len=0, device_len=9, uid=101)
    env["dm"].running_reqs.update([r1, r2])

    pre = _snapshot(env)
    assert pre["running_len"] == 2
    assert pre["free_pages"] < baseline["free_pages"]

    env["sched"]._drain_requests()

    after = _snapshot(env)
    assert after["pending_len"] == 0
    assert after["running_len"] == 0
    assert after["free_pages"] == baseline["free_pages"]
    assert after["table_free_sorted"] == baseline["table_free_sorted"]
    env["cm"].check_integrity()


# =========================================================================
# D. Mixed: pending + chunked + running — everything released.
# =========================================================================


def test_D_mixed_all_resources_reclaimed(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    # 2 pure waiting (no chunked_req, no resources)
    env["pm"].pending_list.append(_make_pending(env, uid=200, input_len=3))
    env["pm"].pending_list.append(_make_pending(env, uid=201, input_len=4))

    # 1 pending with a ChunkedReq occupying a table slot + pages
    ck = _make_req(env, cls=env["ChunkedReq"], cached_len=0, device_len=5, uid=202)
    env["pm"].pending_list.append(
        _make_pending(env, uid=202, input_len=10, chunked_req=ck)
    )

    # 2 running decode reqs
    r1 = _make_req(env, cls=env["Req"], cached_len=0, device_len=7, uid=203)
    r2 = _make_req(env, cls=env["Req"], cached_len=0, device_len=9, uid=204)
    env["dm"].running_reqs.update([r1, r2])

    pre = _snapshot(env)
    assert pre["pending_len"] == 3
    assert pre["running_len"] == 2
    assert pre["free_pages"] < baseline["free_pages"]
    assert pre["table_free_sorted"] != baseline["table_free_sorted"]

    env["sched"]._drain_requests()

    after = _snapshot(env)
    assert after["pending_len"] == 0
    assert after["running_len"] == 0
    assert after["free_pages"] == baseline["free_pages"], (
        f"page leak: {baseline['free_pages']} -> {after['free_pages']}"
    )
    assert after["table_free_sorted"] == baseline["table_free_sorted"], (
        f"table_free drift: {after['table_free_sorted']}"
    )
    env["cm"].check_integrity()


# =========================================================================
# E. Same Req referenced twice — freed exactly once (no double free).
# =========================================================================


def test_E_shared_req_freed_once(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    # A single Req referenced by BOTH the pending list (as a ChunkedReq) AND
    # the running set. This simulates a bookkeeping edge case where the same
    # request object is somehow tracked in two containers.
    shared = _make_req(env, cls=env["ChunkedReq"], cached_len=0, device_len=5, uid=300)
    env["pm"].pending_list.append(
        _make_pending(env, uid=300, input_len=8, chunked_req=shared)
    )
    env["dm"].running_reqs.add(shared)

    env["sched"]._drain_requests()

    after = _snapshot(env)
    # Exactly one table slot restored (not two): if we double-freed, table_free
    # would carry a duplicate entry.
    assert sorted(env["tm"]._free_slots) == baseline["table_free_sorted"]
    assert len(env["tm"]._free_slots) == len(set(env["tm"]._free_slots)), (
        "double-free detected in table_manager._free_slots"
    )
    # No double-freed KV pages either.
    slots = after["free_slots_sorted"]
    assert len(slots) == len(set(slots)), (
        f"double-free detected in cache_manager.free_slots: {slots}"
    )
    assert after["free_pages"] == baseline["free_pages"]
    env["cm"].check_integrity()


# =========================================================================
# F. drain twice — second call is a no-op.
# =========================================================================


def test_F_drain_is_idempotent(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=400)
    env["dm"].running_reqs.add(r)
    env["pm"].pending_list.append(_make_pending(env, uid=401, input_len=4))

    env["sched"]._drain_requests()
    mid = _snapshot(env)
    assert mid["pending_len"] == 0
    assert mid["running_len"] == 0
    assert mid["free_pages"] == baseline["free_pages"]
    assert mid["table_free_sorted"] == baseline["table_free_sorted"]

    # Second call — must not touch allocator state.
    env["sched"]._drain_requests()
    end = _snapshot(env)
    assert end == mid, f"drain #2 mutated state: {mid} -> {end}"


# =========================================================================
# G. Empty scheduler — shutdown drain is a clean no-op.
# =========================================================================


def test_G_empty_scheduler_drain_no_op(sched_env):
    env = sched_env
    baseline = _snapshot(env)

    env["sched"]._drain_requests()

    after = _snapshot(env)
    assert after == baseline
    env["cm"].check_integrity()


# =========================================================================
# H. shutdown() invokes drain in the correct order.
# =========================================================================


def test_H_shutdown_calls_drain_before_engine_shutdown(sched_env):
    env = sched_env
    # Stash real dependencies + monkeypatch shutdown-side collaborators.
    sched = env["sched"]

    call_order: List[str] = []

    from minisgl.utils import device_runtime

    real_sync = device_runtime.synchronize_device
    device_runtime.synchronize_device = lambda dt: call_order.append("sync_device")
    # Rebind the name imported inside scheduler.py too.
    import minisgl.scheduler.scheduler as sched_mod

    sched_mod.synchronize_device = lambda dt: call_order.append("sync_device")

    sched.device_type = "cpu"
    sched.sync_all_ranks = lambda: call_order.append("sync_ranks")

    original_drain = sched._drain_requests

    def _tracked_drain():
        call_order.append("drain")
        return original_drain()

    sched._drain_requests = _tracked_drain

    engine = MagicMock()
    engine.shutdown = lambda: call_order.append("engine_shutdown")
    sched.engine = engine

    # Put one running req so drain has something to do.
    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=500)
    env["dm"].running_reqs.add(r)

    try:
        sched.shutdown()
    finally:
        device_runtime.synchronize_device = real_sync
        sched_mod.synchronize_device = real_sync

    assert call_order == [
        "sync_device",
        "sync_ranks",
        "drain",
        "engine_shutdown",
    ], f"shutdown order wrong: {call_order}"
