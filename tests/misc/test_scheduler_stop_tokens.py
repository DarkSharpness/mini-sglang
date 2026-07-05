"""Gate 2.1c — Scheduler-side stop_token_ids finish contract.

Two-layer test strategy:

1. Structural (AST) — the finish path in ``_process_last_data`` must literally
   consult ``req.sampling_params.stop_token_ids``. This locks the semantics
   even on hosts where the runtime probe cannot execute (no NPU).

2. Behavioural — drive ``_process_last_data`` against a minimal fake
   scheduler + fake batch, exercising the six behaviour rows the Gate 2.1c
   spec calls out:
      a. default empty tuple = no change from pre-2.1c behaviour;
      b. hit on an explicit stop token finishes the request;
      c. the stop token itself is appended to req.input_ids (retained);
      d. ignore_eos=True still honours explicit stop tokens;
      e. non-hit next_token keeps the request running;
      f. max_tokens (can_decode-False) and scalar EOS paths are unchanged.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import List

import torch

from minisgl.core import SamplingParams
from minisgl.message import DetokenizeMsg
from minisgl.scheduler.scheduler import Scheduler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHED_PATH = _REPO_ROOT / "python" / "minisgl" / "scheduler" / "scheduler.py"


# ---------------------------------------------------------------- AST checks


def _process_last_data_fn() -> ast.FunctionDef:
    tree = ast.parse(_SCHED_PATH.read_text())
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    return next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_process_last_data"
    )


def _walk(node):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk(child)


def test_process_last_data_references_stop_token_ids() -> None:
    fn = _process_last_data_fn()
    hits = [
        n for n in _walk(fn)
        if isinstance(n, ast.Attribute) and n.attr == "stop_token_ids"
    ]
    assert hits, (
        "Scheduler._process_last_data must reference sampling_params.stop_token_ids"
    )


def test_stop_token_check_is_in_membership_form() -> None:
    """Must be an ``in`` compare — a plain equality check would not honour
    the tuple contract when multiple stop ids are configured."""
    fn = _process_last_data_fn()
    for cmp_node in [n for n in _walk(fn) if isinstance(n, ast.Compare)]:
        if len(cmp_node.ops) != 1 or not isinstance(cmp_node.ops[0], ast.In):
            continue
        right = cmp_node.comparators[0]
        if (
            isinstance(right, ast.Attribute)
            and right.attr == "stop_token_ids"
        ):
            return
    raise AssertionError(
        "Scheduler._process_last_data must check `next_token in "
        "req.sampling_params.stop_token_ids`"
    )


def test_stop_token_check_does_not_gate_on_ignore_eos() -> None:
    """The stop-token compare must not sit inside an `if not ignore_eos`
    branch — ignore_eos governs only the EOS check, not user stop tokens.

    We look at every ``In`` compare referencing stop_token_ids and walk its
    ancestor If nodes; none of them may test ``ignore_eos``.
    """
    fn = _process_last_data_fn()

    parent = {}
    for n in _walk(fn):
        for c in ast.iter_child_nodes(n):
            parent[c] = n

    def _walk_up(n):
        while n in parent:
            n = parent[n]
            yield n

    stop_cmps = []
    for cmp_node in [n for n in _walk(fn) if isinstance(n, ast.Compare)]:
        if len(cmp_node.ops) == 1 and isinstance(cmp_node.ops[0], ast.In):
            right = cmp_node.comparators[0]
            if isinstance(right, ast.Attribute) and right.attr == "stop_token_ids":
                stop_cmps.append(cmp_node)
    assert stop_cmps, "structural precondition: stop_token_ids compare must exist"

    for cmp_node in stop_cmps:
        for anc in _walk_up(cmp_node):
            if not isinstance(anc, ast.If):
                continue
            names = [a.attr for a in _walk(anc.test) if isinstance(a, ast.Attribute)]
            assert "ignore_eos" not in names, (
                "stop_token_ids check must NOT be nested inside an ignore_eos guard"
            )


# ------------------------------------------------------------- behaviour harness


class _FakeChunkedReq:  # sentinel — the guard on line 160 uses isinstance
    """Stand-in for prefill.ChunkedReq — only the type identity is needed here."""


class _FakeReq:
    def __init__(
        self,
        *,
        input_ids: List[int],
        remain_after_append: int,
        sampling_params: SamplingParams,
        uid: int = 42,
    ) -> None:
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32)
        self._remain_after_append = remain_after_append
        self.sampling_params = sampling_params
        self.uid = uid
        self.append_calls: List[int] = []
        self.free_calls = 0

    def append_host(self, next_token: torch.Tensor) -> None:
        self.append_calls.append(int(next_token.item()))
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self._remain_after_append > 0


class _FakeBatch:
    def __init__(self, reqs, is_prefill: bool = False) -> None:
        self.reqs = reqs
        self.is_prefill = is_prefill


class _FakeCopyEvent:
    def synchronize(self) -> None:
        return None


class _FakeCacheMgr:
    def __init__(self) -> None:
        self.cache_calls: List[tuple] = []

    @contextmanager
    def lazy_free_region(self):
        yield

    def cache_req(self, req, *, finished: bool) -> None:
        self.cache_calls.append((req, finished))


class _FakeDecodeMgr:
    def __init__(self) -> None:
        self.removed: List = []

    def remove_req(self, req) -> None:
        self.removed.append(req)


class _FakeTableMgr:
    def __init__(self) -> None:
        self.freed: List = []

    def free(self, slot) -> None:
        self.freed.append(slot)


def _build_scheduler(reqs, is_prefill=False, eos_token_id: int = 99999):
    """Assemble the minimal Scheduler surface `_process_last_data` touches."""
    sched = Scheduler.__new__(Scheduler)  # bypass __init__ — no engine needed
    sched.eos_token_id = eos_token_id
    sched.cache_manager = _FakeCacheMgr()
    sched.decode_manager = _FakeDecodeMgr()
    sched.table_manager = _FakeTableMgr()
    sched.finished_reqs = set()
    sent: List[List[DetokenizeMsg]] = []
    sched.send_result = lambda reply: sent.append(list(reply))

    def _free_req_resources(req):
        sched.table_manager.free(getattr(req, "table_idx", None))
        sched.cache_manager.cache_req(req, finished=True)

    sched._free_req_resources = _free_req_resources

    # last_data[0] only needs a .batch attribute; last_data[1] must unpack as
    # (_, next_tokens_cpu, copy_done). See Scheduler._process_last_data line 154.
    class _FI:
        pass

    fi = _FI()
    fi.batch = _FakeBatch(reqs, is_prefill=is_prefill)

    next_tokens_cpu = torch.tensor(
        [_pick_next_token(r) for r in reqs], dtype=torch.int32
    )
    output = (None, next_tokens_cpu, _FakeCopyEvent())
    return sched, (fi, output), sent


_NEXT_TOKEN_SLOT: dict = {}


def _pick_next_token(req):
    """Read the pre-programmed next_token for this req."""
    return _NEXT_TOKEN_SLOT[id(req)]


def _program_next_token(req, tok: int) -> None:
    _NEXT_TOKEN_SLOT[id(req)] = tok


# -------------------------------------------------------------------- rows


def test_default_empty_tuple_matches_pre_2_1c_behavior() -> None:
    """Default `stop_token_ids=()` must not cause any premature finish."""
    sp = SamplingParams(max_tokens=8)
    req = _FakeReq(input_ids=[3, 7], remain_after_append=5, sampling_params=sp)
    _program_next_token(req, 42)
    sched, last_data, sent = _build_scheduler([req])
    Scheduler._process_last_data(sched, last_data)
    assert len(sent) == 1 and len(sent[0]) == 1
    msg = sent[0][0]
    assert msg.next_token == 42
    assert msg.finished is False
    assert req.append_calls == [42]
    # Not finished ⇒ no table free, no cache_req (batch is decode-flavored).
    assert sched.table_manager.freed == []
    assert sched.cache_manager.cache_calls == []


def test_stop_token_hit_finishes_and_retains_token() -> None:
    sp = SamplingParams(max_tokens=8, stop_token_ids=(11, 200))
    # remain_after_append=3 -> can_decode=True -> only stop_token_ids can end this.
    req = _FakeReq(input_ids=[3, 7], remain_after_append=3, sampling_params=sp)
    _program_next_token(req, 11)
    sched, last_data, sent = _build_scheduler([req])
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.next_token == 11
    assert msg.finished is True
    # The stop token was appended BEFORE the finish decision.
    assert req.append_calls == [11]
    assert req.input_ids.tolist()[-1] == 11
    # Finished ⇒ scheduler frees the request.
    assert sched.table_manager.freed == [None]  # table_idx wasn't set on fake
    assert sched.cache_manager.cache_calls[0][1] is True  # finished=True


def test_ignore_eos_true_still_honours_stop_token() -> None:
    sp = SamplingParams(
        max_tokens=8, ignore_eos=True, stop_token_ids=(11,)
    )
    req = _FakeReq(input_ids=[3, 7], remain_after_append=3, sampling_params=sp)
    _program_next_token(req, 11)
    sched, last_data, sent = _build_scheduler([req], eos_token_id=99999)
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.next_token == 11
    assert msg.finished is True, (
        "ignore_eos must silence only the tokenizer EOS check; explicit "
        "stop_token_ids must still take effect"
    )


def test_ignore_eos_true_without_stop_token_never_finishes_on_eos_token() -> None:
    """Regression guard: ignore_eos=True must still suppress the scalar EOS."""
    sp = SamplingParams(max_tokens=8, ignore_eos=True, stop_token_ids=())
    req = _FakeReq(input_ids=[3, 7], remain_after_append=3, sampling_params=sp)
    _program_next_token(req, 99999)  # same value as fake eos_token_id below
    sched, last_data, sent = _build_scheduler([req], eos_token_id=99999)
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.next_token == 99999
    assert msg.finished is False


def test_non_matching_next_token_keeps_running() -> None:
    sp = SamplingParams(max_tokens=8, stop_token_ids=(11,))
    req = _FakeReq(input_ids=[3, 7], remain_after_append=3, sampling_params=sp)
    _program_next_token(req, 12)  # not in stop set
    sched, last_data, sent = _build_scheduler([req])
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.finished is False
    assert msg.next_token == 12
    assert sched.table_manager.freed == []


def test_max_tokens_still_finishes_regardless_of_stop_tokens() -> None:
    """remain_after_append==0 -> can_decode=False -> finished True even when
    stop_token_ids is empty."""
    sp = SamplingParams(max_tokens=1, stop_token_ids=())
    req = _FakeReq(input_ids=[3, 7], remain_after_append=0, sampling_params=sp)
    _program_next_token(req, 42)
    sched, last_data, sent = _build_scheduler([req])
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.finished is True
    assert msg.next_token == 42


def test_scalar_eos_still_finishes_when_ignore_eos_false() -> None:
    """The pre-2.1c EOS path must remain intact."""
    sp = SamplingParams(max_tokens=8, ignore_eos=False, stop_token_ids=())
    req = _FakeReq(input_ids=[3, 7], remain_after_append=3, sampling_params=sp)
    _program_next_token(req, 12345)
    sched, last_data, sent = _build_scheduler([req], eos_token_id=12345)
    Scheduler._process_last_data(sched, last_data)
    msg = sent[0][0]
    assert msg.finished is True
    assert msg.next_token == 12345
