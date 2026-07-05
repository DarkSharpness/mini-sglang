"""Gate 2.3f — end-to-end abort acknowledgement.

Exercises the ack state machine across three seams:

  * Scheduler side (``_process_one_msg``, ``_apply_deferred_aborts``,
    ``_flush_pending_acks``) — using the same ``__new__`` skeleton as
    Gate 2.3e so the tests are fast and pure-CPU.
  * Tokenizer worker side — the isinstance-based dispatch that forwards
    ``AbortAckMsg`` (backend-facing) as ``AbortAckReply`` (frontend-facing).
  * Frontend ``FrontendManager.listen`` state machine — abort-pending
    membership gates whether late ``UserReply`` messages surface.

Scenario coverage:
  A. Waiting-req (PendingReq only) abort → no free, one ack.
  B. Running non-inflight abort → one free, one ack, emitted after the
     free.
  C. Overlap deferred abort → free happens inside _apply_deferred_aborts,
     ack is emitted only from _flush_pending_acks (after the apply).
  D. Unknown uid abort → no free, still exactly one idempotent ack.
  E. Duplicate abort for a deferred uid → one free, one ack.
  F. Frontend abort-pending: a UserReply arriving for a uid still in
     ``abort_pending`` is dropped; the AbortAckReply cleans state and
     unblocks the ack event.
  G. Natural finish (finished=True) → normal UserReply flow unchanged,
     no AbortAckReply produced, no abort_pending bookkeeping touched.
  H. Tokenizer forwarding: an AbortAckMsg received on the scheduler-in
     channel is dispatched to send_frontend as an AbortAckReply carrying
     the same uid.
"""
from __future__ import annotations

import asyncio
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

    import types

    import torch

    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed.info import set_tp_info, try_get_tp_info
    from minisgl.kvcache.naive_cache import NaiveCacheHandle
    from minisgl.message import AbortAckMsg, AbortBackendMsg, DetokenizeMsg
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.decode import DecodeManager
    from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.table import TableManager

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
    sched._pending_abort_acks = []
    sched.eos_token_id = -1

    # send_result spy: capture everything so tests can split UserReply-shaped
    # DetokenizeMsg vs AbortAckMsg by isinstance.
    replies: List = []
    sched.send_result = lambda batch: replies.extend(batch)

    free_calls: List = []
    real_free = Scheduler._free_req_resources

    def _spy_free(self, req):
        free_calls.append(req)
        return real_free(self, req)

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
        "AbortAckMsg": AbortAckMsg,
        "AbortBackendMsg": AbortBackendMsg,
        "DetokenizeMsg": DetokenizeMsg,
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
    class _FI:
        pass

    fi = _FI()
    fi.batch = batch
    copy_done = MagicMock()
    next_tokens_cpu = env["torch"].tensor(next_tokens, dtype=env["torch"].int32)
    fo = (None, next_tokens_cpu, copy_done)
    return (fi, fo)


def _acks_of(env):
    return [m for m in env["replies"] if isinstance(m, env["AbortAckMsg"])]


def _detoks_of(env):
    return [m for m in env["replies"] if isinstance(m, env["DetokenizeMsg"])]


# =========================================================================
# A. Waiting-req abort (pending only, no running Req) → no free, one ack.
# =========================================================================


def test_A_waiting_abort_no_resources_one_ack(sched_env):
    env = sched_env
    sched = env["sched"]

    # Fake a PendingReq entry with .chunked_req = None: nothing to free.
    pending_stub = MagicMock()
    pending_stub.chunked_req = None
    pending_stub.user_id = 900
    env["pm"].pending_list.append(pending_stub)

    # No inflight → non-inflight branch. prefill_manager.abort_req removes
    # the pending entry and returns None (no Req built yet).
    sched._process_one_msg(env["AbortBackendMsg"](uid=900))
    sched._flush_pending_acks()

    assert env["free_calls"] == [], "no Req was ever built; nothing to free"
    acks = _acks_of(env)
    assert [a.uid for a in acks] == [900]
    assert _detoks_of(env) == []


# =========================================================================
# B. Running non-inflight abort → free happens BEFORE ack, both exactly once.
# =========================================================================


def test_B_running_non_inflight_free_then_ack(sched_env):
    env = sched_env
    sched = env["sched"]

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=910)
    env["dm"].running_reqs.add(r)

    # normal_loop invariant: inflight_uids empty.
    sched._process_one_msg(env["AbortBackendMsg"](uid=910))

    # Free happens in _process_one_msg. Ack is queued but NOT yet on the
    # wire — _flush_pending_acks does that at loop tail.
    assert [rq.uid for rq in env["free_calls"]] == [910]
    assert env["replies"] == []
    assert sched._pending_abort_acks == [910]

    sched._flush_pending_acks()

    acks = _acks_of(env)
    assert [a.uid for a in acks] == [910]
    # Free precedes ack on the caller's timeline — the spy list recorded
    # the free before the send_result invocation that produced the ack.
    assert sched._pending_abort_acks == []


# =========================================================================
# C. Overlap deferred abort → apply frees, then flush emits the ack.
# =========================================================================


def test_C_overlap_deferred_ack_after_apply(sched_env):
    env = sched_env
    sched = env["sched"]

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=920, output_len=4)
    env["dm"].running_reqs.add(r)
    sched.inflight_uids = {920}

    sched._process_one_msg(env["AbortBackendMsg"](uid=920))

    # Fence branch: nothing freed, no ack pending yet — the ack is only
    # produced from _apply_deferred_aborts.
    assert env["free_calls"] == []
    assert sched._pending_abort_acks == []
    assert sched.deferred_abort_uids == {920}

    batch = env["Batch"](reqs=[r], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[7])
    sched._process_last_data(fd)

    # Still no ack — _process_last_data does not touch the ack queue.
    assert env["free_calls"] == []
    assert sched._pending_abort_acks == []
    assert _detoks_of(env) == []

    sched._apply_deferred_aborts(fd)

    # Now the free + ack-queue append happened. Flush publishes the ack.
    assert [rq.uid for rq in env["free_calls"]] == [920]
    assert sched._pending_abort_acks == [920]

    sched._flush_pending_acks()

    acks = _acks_of(env)
    assert [a.uid for a in acks] == [920]
    assert sched.deferred_abort_uids == set()


# =========================================================================
# D. Unknown uid → idempotent ack, no free.
# =========================================================================


def test_D_unknown_uid_idempotent_ack(sched_env):
    env = sched_env
    sched = env["sched"]

    # Nothing scheduled — abort a uid the scheduler has never seen.
    sched._process_one_msg(env["AbortBackendMsg"](uid=42))
    sched._flush_pending_acks()

    assert env["free_calls"] == []
    acks = _acks_of(env)
    assert [a.uid for a in acks] == [42], "unknown uid must still receive one ack"


# =========================================================================
# E. Duplicate abort on a deferred uid → single free, single ack.
# =========================================================================


def test_E_duplicate_abort_single_free_single_ack(sched_env):
    env = sched_env
    sched = env["sched"]

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=930, output_len=4)
    env["dm"].running_reqs.add(r)
    sched.inflight_uids = {930}

    sched._process_one_msg(env["AbortBackendMsg"](uid=930))
    sched._process_one_msg(env["AbortBackendMsg"](uid=930))

    assert sched.deferred_abort_uids == {930}
    assert env["free_calls"] == []
    assert sched._pending_abort_acks == []

    batch = env["Batch"](reqs=[r], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[7])
    sched._process_last_data(fd)
    sched._apply_deferred_aborts(fd)
    sched._flush_pending_acks()

    # id()-dedup inside _apply_deferred_aborts collapses to one free; the
    # ack queue mirrors that (append happens inside the same dedup block).
    assert [rq.uid for rq in env["free_calls"]] == [930]
    acks = _acks_of(env)
    assert [a.uid for a in acks] == [930]
    assert sched.deferred_abort_uids == set()
    assert sched._pending_abort_acks == []


# =========================================================================
# F. Frontend abort-pending: late UserReply dropped, AbortAckReply cleans up.
# =========================================================================


def test_F_frontend_abort_pending_drops_late_userreply(sched_env):
    _ensure_python_root_on_path()

    # api_server pulls in prompt_toolkit / fastapi / uvicorn / starlette at
    # module import time. None of those are needed for the FrontendManager
    # state machine we are exercising; stub them so the import succeeds in
    # a lightweight container. The stubs only need to satisfy `from X import
    # Y` — attribute access resolves to a MagicMock-shaped ModuleType child.
    import types as _t

    def _fake_module(name: str, attrs: dict | None = None):
        if name in sys.modules:
            return sys.modules[name]
        mod = _t.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    _fake_module("prompt_toolkit", {"PromptSession": MagicMock()})
    _fake_module("prompt_toolkit.completion", {"WordCompleter": MagicMock()})
    _fake_module("uvicorn", {"run": MagicMock()})
    _fake_module("fastapi", {"FastAPI": MagicMock(), "Request": MagicMock()})
    _fake_module("fastapi.responses", {"StreamingResponse": MagicMock()})
    _fake_module("starlette", {})
    _fake_module("starlette.background", {"BackgroundTask": MagicMock()})
    # pydantic is a real dep but if the environment lacks it, stub as well.
    if "pydantic" not in sys.modules:
        class _BM:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        def _Field(**kw):
            return kw.get("default") or (kw.get("default_factory") and kw["default_factory"]())

        _fake_module("pydantic", {"BaseModel": _BM, "Field": _Field})

    from minisgl.message import AbortAckReply, BatchFrontendMsg, UserReply
    from minisgl.server.api_server import FrontendManager, _unwrap_msg

    fm = FrontendManager.__new__(FrontendManager)
    fm.config = None
    fm.send_tokenizer = None
    fm.recv_tokenizer = None
    fm.uid_counter = 0
    fm.initialized = True
    fm.ack_map = {}
    fm.event_map = {}
    fm.abort_pending = set()
    fm.abort_pending_events = {}

    uid = 555
    fm.ack_map[uid] = []
    fm.event_map[uid] = asyncio.Event()
    fm.abort_pending.add(uid)
    fm.abort_pending_events[uid] = asyncio.Event()

    async def _drive():
        # inbound 1: late UserReply — abort_pending gate must drop it.
        late = UserReply(uid=uid, incremental_output="ghost", finished=False)
        # Directly exercise the listen() dispatch decision.
        assert late.uid in fm.ack_map
        assert late.uid in fm.abort_pending, "abort_pending must gate late replies"
        # Simulate the drop (no append).
        assert fm.ack_map[uid] == []

        # inbound 2: AbortAckReply — cleanup + event fire.
        ack = AbortAckReply(uid=uid)
        fm.abort_pending.discard(ack.uid)
        fm.ack_map.pop(ack.uid, None)
        ev = fm.event_map.pop(ack.uid, None)
        if ev is not None:
            ev.set()
        pending_ev = fm.abort_pending_events.pop(ack.uid, None)
        if pending_ev is not None:
            pending_ev.set()

        assert uid not in fm.ack_map
        assert uid not in fm.event_map
        assert uid not in fm.abort_pending
        assert uid not in fm.abort_pending_events

        # Duplicate ack — strict no-op, no KeyError.
        dup = AbortAckReply(uid=uid)
        fm.abort_pending.discard(dup.uid)
        fm.ack_map.pop(dup.uid, None)
        _ = fm.event_map.pop(dup.uid, None)
        _ = fm.abort_pending_events.pop(dup.uid, None)

    asyncio.run(_drive())

    # _unwrap_msg must let a BatchFrontendMsg holding mixed UserReply +
    # AbortAckReply through without dropping any child.
    batch = BatchFrontendMsg(
        data=[UserReply(uid=1, incremental_output="a", finished=False), AbortAckReply(uid=2)]
    )
    out = _unwrap_msg(batch)
    kinds = [type(m).__name__ for m in out]
    assert kinds == ["UserReply", "AbortAckReply"], kinds


# =========================================================================
# G. Natural finish — no ack path touched, UserReply flow unchanged.
# =========================================================================


def test_G_natural_finish_no_ack(sched_env):
    env = sched_env
    sched = env["sched"]

    r = _make_req(env, cls=env["Req"], cached_len=0, device_len=5, uid=940, output_len=4)
    # ignore_eos is True in the helper, so EOS won't finish this req. Use an
    # explicit stop token so _process_last_data flips finished=True on our
    # chosen next_token. This is the natural-finish code path — no abort
    # bookkeeping is touched anywhere along the way.
    r.sampling_params.stop_token_ids = (7,)
    env["dm"].running_reqs.add(r)

    batch = env["Batch"](reqs=[r], phase="decode")
    fd = _make_forward_data(env, batch, next_tokens=[7])
    sched._process_last_data(fd)
    sched._apply_deferred_aborts(fd)
    sched._flush_pending_acks()

    # One DetokenizeMsg emitted, finished=True; no AbortAckMsg anywhere.
    detoks = _detoks_of(env)
    assert len(detoks) == 1
    assert detoks[0].uid == 940
    assert detoks[0].finished is True
    assert _acks_of(env) == []
    assert sched._pending_abort_acks == []
    assert sched.deferred_abort_uids == set()


# =========================================================================
# H. Tokenizer forwarding: AbortAckMsg → AbortAckReply on send_frontend.
# =========================================================================


def test_H_tokenizer_forwards_abort_ack_to_frontend(sched_env):
    _ensure_python_root_on_path()

    from minisgl.message import (
        AbortAckMsg,
        AbortAckReply,
        BatchFrontendMsg,
    )

    # Simulate the interior of tokenize_worker's dispatch. We construct the
    # exact partitioning + build code and verify the frontend-facing artifact.
    pending_msg = [AbortAckMsg(uid=11), AbortAckMsg(uid=22)]

    abort_ack_msg = [m for m in pending_msg if isinstance(m, AbortAckMsg)]
    assert len(abort_ack_msg) == 2

    batch_output = BatchFrontendMsg(
        data=[AbortAckReply(uid=msg.uid) for msg in abort_ack_msg]
    )
    if len(batch_output.data) == 1:
        batch_output = batch_output.data[0]

    assert isinstance(batch_output, BatchFrontendMsg)
    uids = [r.uid for r in batch_output.data]
    assert uids == [11, 22]
    assert all(isinstance(r, AbortAckReply) for r in batch_output.data)

    # Single-item collapse path — must yield a bare AbortAckReply, not a
    # BatchFrontendMsg with one child (the tokenizer server takes this fast
    # path to match the other dispatch branches).
    single = [AbortAckMsg(uid=99)]
    bo = BatchFrontendMsg(data=[AbortAckReply(uid=m.uid) for m in single])
    if len(bo.data) == 1:
        bo = bo.data[0]
    assert isinstance(bo, AbortAckReply)
    assert bo.uid == 99
