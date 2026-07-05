"""Gate 2.3e — Scheduler overlap-loop abort fence.

Exercises the fence state machine on a bare-bones Scheduler skeleton
built via ``__new__``. Uses real ``CacheManager`` / ``TableManager`` /
``PrefillManager`` / ``DecodeManager`` / ``Req`` / ``ChunkedReq`` /
``Batch``. The only mocks are:

  * the outer IO seam (``send_result``, ``receive_msg``) — outside the
    fence's scope,
  * the device-side sampled-token stream (a plain CPU tensor + a
    stub ``copy_done`` event) — same reason,
  * ``_free_req_resources`` is wrapped by a spy that records every
    invocation. The underlying allocator dance (table return + KV
    ``cache_req``) is exercised end-to-end by Gate 2.3d's drain suite;
    here we assert only that the fence produces the correct **number**
    of frees per uid — that is what Gate 2.3e is responsible for.

Scenario coverage:
  A. Ordinary non-inflight abort → immediate free (fence is a no-op).
  B. Inflight abort → resources alive until _process_last_data
     completes; then _apply_deferred_aborts frees exactly once.
  C. Deferred uid does not re-enter the next scheduled batch.
  D. No stale DetokenizeMsg is emitted for the deferred uid.
  E. Two abort msgs for the same inflight uid → still one free.
  F. B=2 batch, one uid aborted mid-fence → the other keeps its token.
  G. normal_loop code path (empty inflight_uids) is unaffected — the
     abort msg takes the immediate-free branch.
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

    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed.info import set_tp_info, try_get_tp_info
    from minisgl.kvcache.naive_cache import NaiveCacheHandle
    from minisgl.message import AbortBackendMsg
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.decode import DecodeManager
    from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.table import TableManager

    # logger.debug_rank0 (invoked from Scheduler._process_one_msg) requires
    # tp_info to be initialised. Every fixture spin-up re-imports minisgl
    # from scratch (see _purge_minisgl_from_sys_modules), so this call is
    # always the first tp_info setter and safe.
    if try_get_tp_info() is None:
        set_tp_info(0, 1)

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
    sched.inflight_uids = set()
    sched.deferred_abort_uids = set()
    # Gate 2.3f: the abort-msg handler unconditionally appends the uid to
    # this list when the non-inflight branch fires (immediate free) OR
    # inside _apply_deferred_aborts. Gate 2.3e tests focus on the fence
    # logic and ignore the ack queue, but the attribute must exist so the
    # scheduler methods do not AttributeError.
    sched._pending_abort_acks = []
    sched.eos_token_id = -1

    # send_result spy: capture emitted DetokenizeMsgs.
    replies: List = []
    sched.send_result = lambda batch: replies.extend(batch)

    # _free_req_resources spy: record every call so tests can assert per-req
    # exact-once free semantics without depending on allocator arithmetic.
    free_calls: List = []
    real_free = Scheduler._free_req_resources

    def _spy_free(self, req):
        free_calls.append(req)
        return real_free(self, req)

    # Bind as a bound method on the instance so calls from any code path
    # (immediate-free branch, _apply_deferred_aborts) route through the spy.
    import types
    sched._free_req_resources = types.MethodType(_spy_free, sched)

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
        "Batch": Batch,
        "SamplingParams": SamplingParams,
        "NaiveCacheHandle": NaiveCacheHandle,
        "AbortBackendMsg": AbortBackendMsg,
        "replies": replies,
        "free_calls": free_calls,
    }


def _make_req(env, *, cls, cached_len: int, device_len: int, uid: int, output_len: int = 8):
    tm = env["tm"]
    cm = env["cm"]
    table_idx = tm.allocate()
    reqs = [
        cls(
            input_ids=env["torch"].zeros(device_len, dtype=env["torch"].int32),
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=output_len,
            uid=uid,
            sampling_params=env["SamplingParams"](
                temperature=0.0, max_tokens=output_len, ignore_eos=True
            ),
            cache_handle=env["NaiveCacheHandle"](),
        )
    ]
    cm.allocate_paged(reqs)
    return reqs[0]


def _make_forward_data(env, batch, next_tokens: List[int]):
    """Build a ``ForwardData`` = (ForwardInput-lite, ForwardOutput-lite).

    ``_process_last_data`` only reads ``last_data[0].batch`` and
    ``last_data[1] = (_, next_tokens_cpu, copy_done)``. It does not
    touch the other ForwardInput fields, so a plain object with a
    ``.batch`` attribute is all the code sees.
    """
    class _FI:
        pass

    fi = _FI()
    fi.batch = batch
    copy_done = MagicMock()
    next_tokens_cpu = env["torch"].tensor(next_tokens, dtype=env["torch"].int32)
    fo = (None, next_tokens_cpu, copy_done)
    return (fi, fo)


# =========================================================================
# A. Non-inflight abort → immediate free (fence branch NOT taken).
# =========================================================================


def test_A_non_inflight_abort_immediate_free(sched_env):
    env = sched_env

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=1)
    env["dm"].running_reqs.add(r)
    assert env["sched"].inflight_uids == set()

    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=1))

    # Immediate-free branch fires: the req is out of running_reqs and
    # _free_req_resources ran exactly once for its uid; no deferred state.
    assert env["dm"].running_reqs == set()
    assert env["sched"].deferred_abort_uids == set()
    assert [rq.uid for rq in env["free_calls"]] == [1], (
        f"immediate abort should free exactly once, got {env['free_calls']}"
    )


# =========================================================================
# B. Inflight abort deferred; resources untouched until apply helper runs.
# =========================================================================


def test_B_inflight_abort_defers_then_frees_once(sched_env):
    env = sched_env

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=42, output_len=4)
    env["dm"].running_reqs.add(r)

    # Simulate overlap_loop entry: this uid is in-flight (batch already
    # forwarded, CPU-side post-process not yet done).
    env["sched"].inflight_uids = {42}

    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=42))

    # Deferred: no free yet. Removed from decode_manager so it cannot
    # re-enter the next batch.
    assert env["sched"].deferred_abort_uids == {42}
    assert env["dm"].running_reqs == set()
    assert env["free_calls"] == [], "resources freed before last_data drained"

    # Drive the fence tail: _process_last_data on the batch containing r.
    batch = env["Batch"](reqs=[r], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[7])

    env["sched"]._process_last_data(fd)

    # No DetokenizeMsg for the deferred uid, no host-token append.
    assert env["replies"] == []
    assert len(r.input_ids) == 5, "append_host ran on a deferred-abort req"
    # Still not freed.
    assert env["free_calls"] == []

    env["sched"]._apply_deferred_aborts(fd)

    # Freed exactly once, deferred set cleared.
    assert [rq.uid for rq in env["free_calls"]] == [42]
    assert env["sched"].deferred_abort_uids == set()


# =========================================================================
# C. Deferred uid is not schedulable — running_reqs no longer holds it.
# =========================================================================


def test_C_deferred_uid_not_in_next_batch(sched_env):
    env = sched_env

    r_a = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=100)
    r_b = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=101)
    env["dm"].running_reqs.update([r_a, r_b])

    env["sched"].inflight_uids = {100, 101}
    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=100))

    # uid 100 is out of the schedulable set; 101 remains. The next
    # decode_manager.schedule_next_batch() (not called here) can therefore
    # only pick 101.
    running_uids = {rq.uid for rq in env["dm"].running_reqs}
    assert running_uids == {101}
    assert env["sched"].deferred_abort_uids == {100}
    assert env["free_calls"] == []


# =========================================================================
# D. Stale DetokenizeMsg suppressed for the deferred uid; peer still replies.
# =========================================================================


def test_D_no_stale_detokenize_for_deferred(sched_env):
    env = sched_env

    r_ok = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=200, output_len=4)
    r_abt = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=201, output_len=4)
    env["dm"].running_reqs.update([r_ok, r_abt])

    env["sched"].inflight_uids = {200, 201}
    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=201))

    batch = env["Batch"](reqs=[r_ok, r_abt], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[10, 20])

    env["sched"]._process_last_data(fd)

    reply_uids = [m.uid for m in env["replies"]]
    assert reply_uids == [200], f"stale reply emitted: {reply_uids}"
    # Surviving req got its token appended.
    assert len(r_ok.input_ids) == 6
    # Aborted req's host buffer must NOT be touched.
    assert len(r_abt.input_ids) == 5

    env["sched"]._apply_deferred_aborts(fd)

    # Aborted uid freed exactly once. r_ok is still alive and was NOT freed
    # here (its lifetime continues into the next tick).
    assert [rq.uid for rq in env["free_calls"]] == [201]


# =========================================================================
# E. Two abort msgs for the same inflight uid → freed exactly once.
# =========================================================================


def test_E_duplicate_inflight_abort_single_free(sched_env):
    env = sched_env

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=300, output_len=4)
    env["dm"].running_reqs.add(r)
    env["sched"].inflight_uids = {300}

    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=300))
    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=300))
    # Set membership guarantees uniqueness — a repeat abort is idempotent.
    assert env["sched"].deferred_abort_uids == {300}
    # No free yet — still fenced.
    assert env["free_calls"] == []

    batch = env["Batch"](reqs=[r], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[9])

    env["sched"]._process_last_data(fd)
    env["sched"]._apply_deferred_aborts(fd)

    # id-dedup inside _apply_deferred_aborts guarantees a single free
    # even if the same Req were somehow to appear twice in batch.reqs.
    assert [rq.uid for rq in env["free_calls"]] == [300]
    # Sanity: no duplicate entries in table_free (would indicate double-free).
    assert len(env["tm"]._free_slots) == len(set(env["tm"]._free_slots))


# =========================================================================
# F. B=2 batch — abort A only, B still gets its token and stays running.
# =========================================================================


def test_F_partial_abort_keeps_survivor(sched_env):
    env = sched_env

    r_a = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=400, output_len=4)
    r_b = _make_req(env, cls=env["Req"], cached_len=0, device_len=6, uid=401, output_len=4)
    env["dm"].running_reqs.update([r_a, r_b])
    env["sched"].inflight_uids = {400, 401}

    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=400))
    # A removed from decode_manager; B remains.
    assert {rq.uid for rq in env["dm"].running_reqs} == {401}

    batch = env["Batch"](reqs=[r_a, r_b], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[999, 555])

    env["sched"]._process_last_data(fd)

    # Exactly one reply — for B, with token 555.
    assert len(env["replies"]) == 1
    assert env["replies"][0].uid == 401
    assert env["replies"][0].next_token == 555
    # B has device_len=6, output_len=4, so max_device_len=10 → remain_len is
    # still positive after append_host; not finished.
    assert env["replies"][0].finished is False
    # B's host token buffer grew by exactly one.
    assert len(r_b.input_ids) == 7
    # A's host buffer untouched.
    assert len(r_a.input_ids) == 5

    env["sched"]._apply_deferred_aborts(fd)

    # After apply: A is freed (once), B remains scheduled and unfreed.
    assert [rq.uid for rq in env["free_calls"]] == [400]
    assert {rq.uid for rq in env["dm"].running_reqs} == {401}
    assert env["sched"].deferred_abort_uids == set()


# =========================================================================
# G. normal_loop path (empty inflight_uids) takes immediate-free branch.
# =========================================================================


def test_G_normal_loop_behavior_unchanged(sched_env):
    env = sched_env

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=500)
    env["dm"].running_reqs.add(r)
    # normal_loop invariant: inflight_uids never populated.
    assert env["sched"].inflight_uids == set()

    env["sched"]._process_one_msg(env["AbortBackendMsg"](uid=500))

    # Immediate-free branch: no deferred bookkeeping, req yanked, freed once.
    assert env["sched"].deferred_abort_uids == set()
    assert env["dm"].running_reqs == set()
    assert [rq.uid for rq in env["free_calls"]] == [500]
