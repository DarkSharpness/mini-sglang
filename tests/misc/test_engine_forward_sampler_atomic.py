"""Gate 2.3c — atomic sampler commit in ``Engine.forward_batch``.

These tests pin the contract: ``req.complete_one()`` only fires when both
``model.forward()`` and ``sampler.sample()`` succeeded AND the sampler
tensor passes a basic shape check. Any failure inside those steps must
leave ``req.device_len`` / ``req.cached_len`` / ``req.extend_len`` at the
values they had on entry — and it must do so uniformly for every real
request in ``batch.reqs`` (batch-level all-or-nothing).

Tests avoid the real Engine constructor (needs weights, CUDA/NPU). They
build a minimal ``Engine`` shell via ``__new__`` and bolt on:
  - a ``graph_runner`` whose ``can_use_cuda_graph`` returns False
  - a ``model`` whose ``forward()`` returns a fixed logits tensor
  - a ``sampler`` whose behaviour each test controls
  - a ``ctx`` that just yields on ``forward_batch``
so ``forward_batch`` runs its own real body.

All requests are real ``Req`` objects — ``complete_one()`` is exercised
for real. The sentinel tokens returned by the successful sampler feed
straight into the ``next_tokens_cpu`` copy, which we assert on.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List

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


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def env():
    _ensure_python_root_on_path()
    _purge_minisgl_from_sys_modules()

    import torch

    from minisgl.core import Req, SamplingParams
    from minisgl.engine.engine import Engine, ForwardOutput
    from minisgl.engine.sample import BatchSamplingArgs
    from minisgl.kvcache.naive_cache import NaiveCacheHandle

    return SimpleNamespace(
        torch=torch,
        Req=Req,
        SamplingParams=SamplingParams,
        Engine=Engine,
        ForwardOutput=ForwardOutput,
        BatchSamplingArgs=BatchSamplingArgs,
        NaiveCacheHandle=NaiveCacheHandle,
    )


def _make_req(env, *, table_idx: int, cached_len: int, device_len: int, uid: int):
    assert 0 <= cached_len < device_len
    input_ids = env.torch.zeros(device_len, dtype=env.torch.int32)
    return env.Req(
        input_ids=input_ids,
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=8,  # generous so remain_len > 0 always
        uid=uid,
        sampling_params=env.SamplingParams(temperature=0.0, max_tokens=1),
        cache_handle=env.NaiveCacheHandle(),
    )


def _snapshot_reqs(reqs) -> List[dict]:
    return [
        {
            "uid": r.uid,
            "table_idx": r.table_idx,
            "cached_len": r.cached_len,
            "device_len": r.device_len,
            "extend_len": r.extend_len,
        }
        for r in reqs
    ]


def _assert_reqs_unchanged(before: List[dict], reqs) -> None:
    after = _snapshot_reqs(reqs)
    assert before == after, f"req state advanced despite sampler failure: {before} -> {after}"


# --------------------------------------------------------- engine skeleton


def _build_engine_skeleton(env, *, logits_shape, sampler_impl):
    """Return an Engine shell that exercises the real ``forward_batch`` body."""
    from minisgl.core import Batch  # noqa: F401  (only imported for clarity)

    engine = env.Engine.__new__(env.Engine)
    device = env.torch.device("cpu")
    engine.device = device
    engine.device_type = "cpu"
    engine.stream = None  # cpu device_runtime returns None

    logits_tensor = env.torch.zeros(logits_shape, dtype=env.torch.float32, device=device)

    # graph_runner: never uses cuda graph, so `.replay` is never called.
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)

    # model.forward returns the fixed logits tensor.
    engine.model = SimpleNamespace(forward=lambda: logits_tensor)

    # sampler.sample: caller-controlled.
    engine.sampler = SimpleNamespace(sample=sampler_impl)

    # ctx.forward_batch: contextmanager that just yields.
    @contextmanager
    def _fake_forward_batch(batch):
        yield

    engine.ctx = SimpleNamespace(forward_batch=_fake_forward_batch)

    return engine


def _run_forward(env, engine, reqs, *, phase="prefill"):
    from minisgl.core import Batch

    batch = Batch(reqs=list(reqs), phase=phase)
    batch.padded_reqs = list(reqs)
    args = env.BatchSamplingArgs(temperatures=None)
    return engine.forward_batch(batch, args)


# =========================================================================
# A. B=1 sampler failure — req state preserved, original exception propagates.
# =========================================================================


def test_A_b1_sampler_failure_preserves_req_state(env):
    reqs = [_make_req(env, table_idx=0, cached_len=0, device_len=4, uid=1)]
    before = _snapshot_reqs(reqs)

    class GateSamplerBoom(Exception):
        pass

    def _explode(logits, args):
        raise GateSamplerBoom("gate23c_A: sampler blew up")

    engine = _build_engine_skeleton(
        env, logits_shape=(1, 16), sampler_impl=_explode
    )

    with pytest.raises(GateSamplerBoom, match="gate23c_A"):
        _run_forward(env, engine, reqs)

    _assert_reqs_unchanged(before, reqs)


# =========================================================================
# B. B=2 sampler failure — no partial commit; both reqs unchanged.
# =========================================================================


def test_B_b2_sampler_failure_no_partial_commit(env):
    reqs = [
        _make_req(env, table_idx=0, cached_len=0, device_len=4, uid=10),
        _make_req(env, table_idx=1, cached_len=2, device_len=6, uid=11),
    ]
    before = _snapshot_reqs(reqs)

    def _explode(logits, args):
        raise RuntimeError("gate23c_B: mid-sampler failure")

    engine = _build_engine_skeleton(
        env, logits_shape=(2, 16), sampler_impl=_explode
    )

    with pytest.raises(RuntimeError, match="gate23c_B"):
        _run_forward(env, engine, reqs)

    _assert_reqs_unchanged(before, reqs)


# =========================================================================
# C. Sampler success — each real request advances exactly once.
# =========================================================================


def test_C_sampler_success_each_req_advances_once(env):
    reqs = [
        _make_req(env, table_idx=0, cached_len=0, device_len=4, uid=20),
        _make_req(env, table_idx=1, cached_len=1, device_len=5, uid=21),
    ]
    before = _snapshot_reqs(reqs)

    def _ok(logits, args):
        # Return deterministic sentinel token per row.
        return env.torch.tensor([42, 43], dtype=env.torch.int64)

    engine = _build_engine_skeleton(env, logits_shape=(2, 16), sampler_impl=_ok)
    out = _run_forward(env, engine, reqs)

    # Each req: cached_len := old_device_len; device_len := old_device_len + 1.
    for b, r in zip(before, reqs):
        assert r.cached_len == b["device_len"], (
            f"cached_len expected {b['device_len']} got {r.cached_len} for uid={b['uid']}"
        )
        assert r.device_len == b["device_len"] + 1, (
            f"device_len expected {b['device_len'] + 1} got {r.device_len} for uid={b['uid']}"
        )
        # extend_len after complete_one is device_len - cached_len == 1.
        assert r.extend_len == 1

    # Sampler tokens flow through the ForwardOutput untouched (converted to int32).
    assert out.next_tokens_cpu.dtype == env.torch.int32
    assert out.next_tokens_cpu.tolist() == [42, 43]


# =========================================================================
# D. Sampler returns wrong batch dim — raise before commit.
# =========================================================================


def test_D_sampler_wrong_shape_no_commit(env):
    reqs = [
        _make_req(env, table_idx=0, cached_len=0, device_len=4, uid=30),
        _make_req(env, table_idx=1, cached_len=0, device_len=4, uid=31),
    ]
    before = _snapshot_reqs(reqs)

    def _bad_shape(logits, args):
        # Only ONE row for a 2-request batch — must be rejected.
        return env.torch.tensor([99], dtype=env.torch.int64)

    engine = _build_engine_skeleton(
        env, logits_shape=(2, 16), sampler_impl=_bad_shape
    )

    with pytest.raises(RuntimeError, match="shape"):
        _run_forward(env, engine, reqs)

    _assert_reqs_unchanged(before, reqs)


# =========================================================================
# E. padded batch — only real reqs get committed.
# =========================================================================


def test_E_padded_batch_only_real_reqs_commit(env):
    # Two real reqs, one padded "extra" appended to padded_reqs only.
    real_reqs = [
        _make_req(env, table_idx=0, cached_len=0, device_len=4, uid=40),
        _make_req(env, table_idx=1, cached_len=2, device_len=6, uid=41),
    ]
    padding_req = _make_req(env, table_idx=2, cached_len=0, device_len=4, uid=999)
    before_real = _snapshot_reqs(real_reqs)
    before_pad = _snapshot_reqs([padding_req])

    def _ok(logits, args):
        # Sampler operates on logits[:batch.size] which is 2 rows — return 2 tokens.
        return env.torch.tensor([7, 8], dtype=env.torch.int64)

    from minisgl.core import Batch

    engine = _build_engine_skeleton(
        env,
        # Logits shape padded to 3 rows (mimics graph capture pad); slicing by
        # batch.size restricts sampler input to the real portion.
        logits_shape=(3, 16),
        sampler_impl=_ok,
    )

    batch = Batch(reqs=list(real_reqs), phase="prefill")
    batch.padded_reqs = list(real_reqs) + [padding_req]
    args = env.BatchSamplingArgs(temperatures=None)
    out = engine.forward_batch(batch, args)

    # Real reqs advanced once each.
    for b, r in zip(before_real, real_reqs):
        assert r.cached_len == b["device_len"]
        assert r.device_len == b["device_len"] + 1

    # Padding req untouched — complete_one iterates batch.reqs only.
    assert _snapshot_reqs([padding_req]) == before_pad

    assert out.next_tokens_cpu.tolist() == [7, 8]
