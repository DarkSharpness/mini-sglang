"""Hermetic tests for ``python/minisgl/layers/attention.py`` AttentionLayer.

No CUDA / NPU hardware and no real ``flashinfer`` / ``torch_npu`` are required.

The tests build a real ``AttentionLayer`` on CPU, then replace ``.rotary`` with
a fake identity callable so we can drive the Q/K/V handoff contract without
depending on the platform RoPE op. RMSNorm and the attention backend are also
fake, purely to record what shapes each stage received.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


MODULE_PATH = "minisgl.layers.attention"
SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "minisgl"
    / "layers"
    / "attention.py"
)


# ============================================================ constants
# ``RotaryEmbedding`` asserts head_size in {64, 128, 256, 512}. We pick 64 so
# the surrounding tests stay small while still constructing a real rotary.
Hq = 4
Hk = 2
D = 64
QKV_LAST = (Hq + 2 * Hk) * D  # 512


# ============================================================ helpers
class _FakeRotary:
    """Records inputs and returns them unchanged (identity)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward(self, positions, query, key):
        self.calls.append(
            {
                "positions_id": id(positions),
                "positions_shape": tuple(positions.shape),
                "positions_dtype": positions.dtype,
                "query_shape": tuple(query.shape),
                "query_dtype": query.dtype,
                "query_contig": query.is_contiguous(),
                "key_shape": tuple(key.shape),
                "key_dtype": key.dtype,
                "key_contig": key.is_contiguous(),
            }
        )
        return query, key


class _FakeNorm:
    """Records incoming shapes, no-op forward_inplace."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward_inplace(self, x):
        self.calls.append(
            {
                "shape": tuple(x.shape),
                "dtype": x.dtype,
                "contig": x.is_contiguous(),
                "data_ptr": x.data_ptr(),
            }
        )
        return x


class _CaptureBackend:
    def __init__(self, return_shape=None):
        self.calls = 0
        self.args: dict[str, Any] | None = None
        self._return_shape = return_shape

    def forward(self, q, k, v, layer_id, batch):
        self.calls += 1
        self.args = {
            "q_shape": tuple(q.shape),
            "q_stride": q.stride(),
            "q_dtype": q.dtype,
            "q_data_ptr": q.data_ptr(),
            "k_shape": tuple(k.shape),
            "k_stride": k.stride(),
            "k_dtype": k.dtype,
            "k_data_ptr": k.data_ptr(),
            "v_shape": tuple(v.shape),
            "v_stride": v.stride(),
            "v_dtype": v.dtype,
            "v_data_ptr": v.data_ptr(),
            "layer_id": layer_id,
            "batch_id": id(batch),
            "positions_id": id(batch.positions),
        }
        shape = self._return_shape or tuple(q.shape)
        return torch.zeros(shape, dtype=q.dtype)


# ============================================================ fixtures
@pytest.fixture()
def clean_tp():
    """Reset TP info so ``set_tp_info(rank=0, size=1)`` succeeds each test."""
    import minisgl.distributed.info as info_mod

    saved = info_mod._TP_INFO
    info_mod._TP_INFO = None
    from minisgl.distributed import set_tp_info

    set_tp_info(rank=0, size=1)
    yield
    info_mod._TP_INFO = saved


@pytest.fixture()
def clean_ctx():
    import minisgl.core as core_mod

    saved = core_mod._GLOBAL_CTX
    core_mod._GLOBAL_CTX = None
    yield
    core_mod._GLOBAL_CTX = saved


@pytest.fixture()
def attn_factory(clean_tp, clean_ctx):
    """Return a factory that builds an AttentionLayer with fake sub-modules."""
    from minisgl.layers.attention import AttentionLayer
    from minisgl.models.config import RotaryConfig
    from minisgl.layers.rotary import set_rope_device

    # RotaryEmbedding's ctor allocates a cos/sin cache on the current default
    # device; the CI host defaults to CPU which is fine, but we make it
    # explicit and small.
    set_rope_device(torch.device("cpu"))

    def _make(*, q_norm=None, k_norm=None):
        rotary_config = RotaryConfig(
            head_dim=D,
            rotary_dim=D,
            max_position=64,
            base=10000.0,
            scaling=None,
        )
        layer = AttentionLayer(
            layer_id=7,
            num_qo_heads=Hq,
            num_kv_heads=Hk,
            head_dim=D,
            rotary_config=rotary_config,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        return layer

    return _make


def _install_ctx(backend, positions):
    from minisgl.core import Context, Batch, set_global_ctx

    ctx = Context(page_size=1)
    ctx.attn_backend = backend
    set_global_ctx(ctx)
    batch = Batch(reqs=[], phase="prefill")
    batch.positions = positions
    return ctx, batch


# ============================================================ 1. shape flow
@pytest.mark.parametrize("T", [1, 6])
def test_qkv_reshape_and_backend_shapes(attn_factory, T):
    q_norm = _FakeNorm()
    k_norm = _FakeNorm()
    layer = attn_factory(q_norm=q_norm, k_norm=k_norm)
    fake_rotary = _FakeRotary()
    layer.rotary = fake_rotary

    backend = _CaptureBackend()
    positions = torch.arange(T, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    torch.manual_seed(101)
    qkv = torch.randn(T, QKV_LAST, dtype=torch.float32)

    with torch.inference_mode():
        with ctx.forward_batch(batch):
            out = layer.forward(qkv)

    assert tuple(out.shape) == (T, Hq * D)

    # q_norm and k_norm each received the 3D reshape
    assert len(q_norm.calls) == 1
    assert q_norm.calls[0]["shape"] == (T, Hq, D)
    assert len(k_norm.calls) == 1
    assert k_norm.calls[0]["shape"] == (T, Hk, D)

    # rotary got 3D q/k
    assert len(fake_rotary.calls) == 1
    rc = fake_rotary.calls[0]
    assert rc["query_shape"] == (T, Hq, D)
    assert rc["key_shape"] == (T, Hk, D)
    assert rc["positions_id"] == id(positions)

    # backend got 3D q/k/v with correct head splits
    assert backend.calls == 1
    assert backend.args is not None
    assert backend.args["q_shape"] == (T, Hq, D)
    assert backend.args["k_shape"] == (T, Hk, D)
    assert backend.args["v_shape"] == (T, Hk, D)
    assert backend.args["layer_id"] == 7
    assert backend.args["batch_id"] == id(batch)
    assert backend.args["positions_id"] == id(positions)


# ============================================================ 2. no-copy reshape
@pytest.mark.parametrize("T", [1, 6])
def test_reshape_does_not_copy(attn_factory, T):
    q_norm = _FakeNorm()
    k_norm = _FakeNorm()
    layer = attn_factory(q_norm=q_norm, k_norm=k_norm)
    fake_rotary = _FakeRotary()
    layer.rotary = fake_rotary
    backend = _CaptureBackend()
    positions = torch.arange(T, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    torch.manual_seed(102)
    qkv = torch.randn(T, QKV_LAST, dtype=torch.float32)
    qkv_ptr = qkv.data_ptr()

    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    # data_ptr of the tensors passed to norms and backend must all fall inside
    # qkv's storage — the split+view chain is a pure view.
    q_ptr = q_norm.calls[0]["data_ptr"]
    k_ptr = k_norm.calls[0]["data_ptr"]
    b_v_ptr = backend.args["v_data_ptr"]

    element_bytes = qkv.element_size()
    q_offset = (q_ptr - qkv_ptr) // element_bytes
    k_offset = (k_ptr - qkv_ptr) // element_bytes
    v_offset = (b_v_ptr - qkv_ptr) // element_bytes

    # Q starts at qkv[..., 0], K at qkv[..., Hq*D], V at qkv[..., (Hq+Hk)*D]
    assert q_offset == 0
    assert k_offset == Hq * D
    assert v_offset == (Hq + Hk) * D


# ============================================================ 3. numerical Q/K/V vs slice-reshape reference
@pytest.mark.parametrize("T", [1, 6])
def test_qkv_values_match_slice_reshape(attn_factory, T):
    layer = attn_factory(q_norm=None, k_norm=None)
    fake_rotary = _FakeRotary()

    captured = {}

    def rotary_capture(positions, q, k):
        captured["q_in"] = q.detach().clone()
        captured["k_in"] = k.detach().clone()
        return q, k

    class _Rot:
        def forward(self, positions, q, k):
            return rotary_capture(positions, q, k)

    layer.rotary = _Rot()

    backend = _CaptureBackend()

    def _bf(q, k, v, layer_id, batch):
        captured["backend_q"] = q.detach().clone()
        captured["backend_k"] = k.detach().clone()
        captured["backend_v"] = v.detach().clone()
        return _CaptureBackend.forward(backend, q, k, v, layer_id, batch)

    backend.forward = _bf  # type: ignore[assignment]

    positions = torch.arange(T, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    torch.manual_seed(203)
    qkv = torch.randn(T, QKV_LAST, dtype=torch.float32)

    # Reference: pure slice + reshape on qkv.
    ref_q = qkv[:, : Hq * D].contiguous().view(T, Hq, D)
    ref_k = qkv[:, Hq * D : (Hq + Hk) * D].contiguous().view(T, Hk, D)
    ref_v = qkv[:, (Hq + Hk) * D :].contiguous().view(T, Hk, D)

    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    assert torch.equal(captured["q_in"], ref_q)
    assert torch.equal(captured["k_in"], ref_k)
    assert torch.equal(captured["backend_v"], ref_v)


# ============================================================ 4. V ordering: catch silent gate/order swaps
def test_v_slice_is_the_last_third(attn_factory):
    """Sentinel test: fill each split with a distinct constant and verify that
    V corresponds to the LAST slice, not Q or K.
    """
    layer = attn_factory(q_norm=None, k_norm=None)
    layer.rotary = _FakeRotary()
    backend = _CaptureBackend()

    captured = {}

    def _bf(q, k, v, layer_id, batch):
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        captured["v"] = v.detach().clone()
        return torch.zeros_like(q)

    backend.forward = _bf  # type: ignore[assignment]

    T = 3
    positions = torch.arange(T, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    qkv = torch.empty(T, QKV_LAST, dtype=torch.float32)
    qkv[:, : Hq * D] = 1.0
    qkv[:, Hq * D : (Hq + Hk) * D] = 2.0
    qkv[:, (Hq + Hk) * D :] = 3.0

    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    assert torch.all(captured["q"] == 1.0)
    assert torch.all(captured["k"] == 2.0)
    assert torch.all(captured["v"] == 3.0)


# ============================================================ 5. no q/k norm branch
@pytest.mark.parametrize("T", [1, 6])
def test_forward_without_norms(attn_factory, T):
    layer = attn_factory(q_norm=None, k_norm=None)
    fake_rotary = _FakeRotary()
    layer.rotary = fake_rotary
    backend = _CaptureBackend()
    positions = torch.arange(T, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    torch.manual_seed(303)
    qkv = torch.randn(T, QKV_LAST, dtype=torch.float32)

    with torch.inference_mode():
        with ctx.forward_batch(batch):
            out = layer.forward(qkv)

    assert tuple(out.shape) == (T, Hq * D)
    assert len(fake_rotary.calls) == 1
    assert fake_rotary.calls[0]["query_shape"] == (T, Hq, D)
    assert fake_rotary.calls[0]["key_shape"] == (T, Hk, D)
    assert backend.calls == 1


# ============================================================ 6. positions and batch passthrough
def test_positions_and_batch_passthrough(attn_factory):
    layer = attn_factory(q_norm=_FakeNorm(), k_norm=_FakeNorm())
    fake_rotary = _FakeRotary()
    layer.rotary = fake_rotary
    backend = _CaptureBackend()
    positions = torch.arange(4, dtype=torch.int64)  # deliberately int64
    ctx, batch = _install_ctx(backend, positions)

    qkv = torch.randn(4, QKV_LAST, dtype=torch.float32)
    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    # rotary saw the same positions object
    assert fake_rotary.calls[0]["positions_id"] == id(positions)
    assert fake_rotary.calls[0]["positions_dtype"] == torch.int64
    # backend saw the same batch object
    assert backend.args["batch_id"] == id(batch)
    # backend saw the same positions via batch
    assert backend.args["positions_id"] == id(positions)


# ============================================================ 7. layer_id passthrough
def test_layer_id_passthrough(attn_factory):
    layer = attn_factory()
    layer.rotary = _FakeRotary()
    backend = _CaptureBackend()
    positions = torch.arange(2, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    qkv = torch.randn(2, QKV_LAST, dtype=torch.float32)
    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    assert backend.args["layer_id"] == 7  # set in the factory


# ============================================================ 8. call counts
def test_call_counts(attn_factory):
    q_norm = _FakeNorm()
    k_norm = _FakeNorm()
    layer = attn_factory(q_norm=q_norm, k_norm=k_norm)
    fake_rotary = _FakeRotary()
    layer.rotary = fake_rotary
    backend = _CaptureBackend()
    positions = torch.arange(5, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    qkv = torch.randn(5, QKV_LAST, dtype=torch.float32)
    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)

    assert len(q_norm.calls) == 1
    assert len(k_norm.calls) == 1
    assert len(fake_rotary.calls) == 1
    assert backend.calls == 1


# ============================================================ 9. no .contiguous() call in source
def test_no_contiguous_call_in_forward():
    src = SOURCE_PATH.read_text()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            fn = node
            break
    assert fn is not None
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Attribute) and sub.attr == "contiguous":
            raise AssertionError(
                "AttentionLayer.forward must not call .contiguous() on Q/K/V"
            )


# ============================================================ 10. bad qkv last dim raises
def test_bad_qkv_last_dim_raises(attn_factory):
    layer = attn_factory()
    layer.rotary = _FakeRotary()
    backend = _CaptureBackend()
    positions = torch.arange(2, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    bad = torch.randn(2, QKV_LAST + 1, dtype=torch.float32)
    with torch.inference_mode():
        with ctx.forward_batch(batch):
            with pytest.raises((RuntimeError, ValueError)):
                layer.forward(bad)


# ============================================================ 11. import guard
def test_module_import_does_not_pull_flashinfer_torch_npu_or_kernel():
    # Fresh import to observe what the module drags in.
    for name in list(sys.modules):
        if name == MODULE_PATH:
            del sys.modules[name]
    for name in ("flashinfer", "torch_npu"):
        sys.modules.pop(name, None)

    importlib.import_module(MODULE_PATH)

    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules
    for name in list(sys.modules):
        assert not name.startswith("minisgl.kernel"), (
            f"AttentionLayer import pulled minisgl.kernel: {name}"
        )


# ============================================================ 12. grad-mode characterization
def test_grad_mode_characterization(attn_factory):
    """Characterization: outside ``inference_mode``, ``forward_inplace`` on a
    view returned by ``torch.split`` may raise. Production runs under
    ``torch.inference_mode()`` (scheduler / server / tokenizer_server) so we
    only record this behaviour here; failure is expected and not asserted.
    """
    q_norm = _FakeNorm()
    k_norm = _FakeNorm()

    # Give the fake norm a real inplace copy_ so grad-mode restriction bites.
    def fi_with_copy(x):
        y = x + 0
        x.copy_(y)
        return x

    q_norm.forward_inplace = fi_with_copy  # type: ignore[assignment]
    k_norm.forward_inplace = fi_with_copy  # type: ignore[assignment]

    layer = attn_factory(q_norm=q_norm, k_norm=k_norm)
    layer.rotary = _FakeRotary()
    backend = _CaptureBackend()
    positions = torch.arange(1, dtype=torch.int32)
    ctx, batch = _install_ctx(backend, positions)

    qkv = torch.randn(1, QKV_LAST, dtype=torch.float32)

    # Grad mode: torch may or may not reject the copy_ on a split-view. Both
    # outcomes are informational; do not assert.
    grad_mode_ok = None
    grad_mode_err = None
    try:
        with ctx.forward_batch(batch):
            layer.forward(qkv)
        grad_mode_ok = True
    except RuntimeError as exc:
        grad_mode_ok = False
        grad_mode_err = str(exc)

    # Inference mode must succeed regardless (this is the production contract).
    inf_ok = False
    with torch.inference_mode():
        with ctx.forward_batch(batch):
            layer.forward(qkv)
        inf_ok = True
    assert inf_ok

    # Emit for record; keep test always green.
    print(f"[gate1.9t] grad_mode_ok={grad_mode_ok}  err={grad_mode_err!r}")
