"""Hermetic tests for ``python/minisgl/layers/norm.py`` device dispatch.

The tests never touch CUDA or NPU hardware. Fake ``flashinfer`` and
``torch_npu`` modules are injected via ``monkeypatch`` and a fake tensor
wrapper carries a controlled ``device.type`` string through the dispatch.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch


# ============================================================ fixtures
@pytest.fixture
def clean_optional_deps(monkeypatch):
    """Drop any previously-cached flashinfer / torch_npu from sys.modules."""
    for name in list(sys.modules):
        head = name.split(".", 1)[0]
        if head in {"flashinfer", "torch_npu"}:
            monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture
def fresh_norm(clean_optional_deps):
    """Re-import ``minisgl.layers.norm`` on a clean slate."""
    monkeypatch_targets = [
        "minisgl.layers.norm",
    ]
    for name in monkeypatch_targets:
        if name in sys.modules:
            del sys.modules[name]
    import minisgl.layers.norm as norm
    return norm


class _BlockingFinder:
    """Meta-path finder that raises ImportError for a fixed set of top names."""

    def __init__(self, *blocked: str) -> None:
        self._blocked = set(blocked)

    def find_spec(self, name, path=None, target=None):
        head = name.split(".", 1)[0]
        if head in self._blocked:
            raise ImportError(f"blocked by test finder: {name}")
        return None


@pytest.fixture
def block_flashinfer(monkeypatch):
    for name in list(sys.modules):
        if name.split(".", 1)[0] == "flashinfer":
            monkeypatch.delitem(sys.modules, name, raising=False)
    finder = _BlockingFinder("flashinfer")
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))


@pytest.fixture
def block_torch_npu(monkeypatch):
    for name in list(sys.modules):
        if name.split(".", 1)[0] == "torch_npu":
            monkeypatch.delitem(sys.modules, name, raising=False)
    finder = _BlockingFinder("torch_npu")
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))


# ============================================================ helpers
class _FakeDevice:
    def __init__(self, t: str) -> None:
        self.type = t

    def __repr__(self) -> str:
        return f"_FakeDevice(type={self.type!r})"


class _FakeTensor:
    """Minimal tensor stand-in.

    The dispatch code only reads ``x.device.type`` and, on the inplace path,
    calls ``x.copy_(y)``. Everything else is forwarded to the underlying real
    tensor when needed.
    """

    def __init__(self, real: torch.Tensor, device_type: str) -> None:
        self._real = real
        self.device = _FakeDevice(device_type)
        self.copy_calls: list = []

    def copy_(self, other):
        self.copy_calls.append(other)
        payload = other._real if isinstance(other, _FakeTensor) else other
        if isinstance(payload, torch.Tensor):
            self._real.copy_(payload)
        return self


def _cpu_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    return (x_f * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _install_fake_torch_npu(monkeypatch, rms_norm_impl=None, add_rms_norm_impl=None):
    fake = types.ModuleType("torch_npu")
    if rms_norm_impl is not None:
        fake.npu_rms_norm = rms_norm_impl
    if add_rms_norm_impl is not None:
        fake.npu_add_rms_norm = add_rms_norm_impl
    monkeypatch.setitem(sys.modules, "torch_npu", fake)


def _install_fake_flashinfer(monkeypatch, rmsnorm_impl=None, fused_add_impl=None):
    fake = types.ModuleType("flashinfer")
    if rmsnorm_impl is not None:
        fake.rmsnorm = rmsnorm_impl
    if fused_add_impl is not None:
        fake.fused_add_rmsnorm = fused_add_impl
    monkeypatch.setitem(sys.modules, "flashinfer", fake)


# ================================================================ #1
def test_import_norm_does_not_trigger_optional_deps(fresh_norm):
    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules


# ================================================================ #2
def test_construct_classes_does_not_import_optional_deps(fresh_norm):
    _ = fresh_norm.RMSNorm(16, 1e-6)
    _ = fresh_norm.RMSNormFused(16, 1e-6)
    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules


# ================================================================ #3
def test_cpu_rmsnorm_forward_matches_reference(fresh_norm):
    torch.manual_seed(0)
    norm = fresh_norm.RMSNorm(64, 1e-6)
    norm.weight = torch.randn(64)
    x = torch.randn(3, 5, 64)

    y = norm.forward(x)
    ref = _cpu_reference(x, norm.weight, norm.eps)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.allclose(y, ref, atol=1e-6)


# ================================================================ #4
def test_cpu_rmsnorm_forward_inplace_mutates_and_returns_none(fresh_norm):
    torch.manual_seed(1)
    norm = fresh_norm.RMSNorm(32, 1e-6)
    norm.weight = torch.randn(32)
    x = torch.randn(4, 32)
    x_orig = x.clone()
    x_id = id(x)
    ptr = x.data_ptr()

    ret = norm.forward_inplace(x)

    assert ret is None
    assert id(x) == x_id
    assert x.data_ptr() == ptr
    ref = _cpu_reference(x_orig, norm.weight, norm.eps)
    assert torch.allclose(x, ref, atol=1e-6)


# ================================================================ #5
def test_cpu_fused_residual_none_passes_original_x(fresh_norm):
    torch.manual_seed(2)
    norm = fresh_norm.RMSNormFused(32, 1e-6)
    norm.weight = torch.randn(32)
    x = torch.randn(3, 32)
    x_orig = x.clone()

    normalized, passthrough = norm.forward(x, residual=None)

    assert passthrough is x
    assert torch.equal(x, x_orig)
    ref = _cpu_reference(x_orig, norm.weight, norm.eps)
    assert torch.allclose(normalized, ref, atol=1e-6)


# ================================================================ #6
def test_cpu_fused_with_residual(fresh_norm):
    torch.manual_seed(3)
    norm = fresh_norm.RMSNormFused(64, 1e-6)
    norm.weight = torch.randn(64)
    x = torch.randn(2, 64)
    residual = torch.randn(2, 64)
    x_orig = x.clone()
    residual_orig = residual.clone()

    normalized, summed = norm.forward(x, residual)

    # CPU + NPU mirror the "no in-place on inputs" contract.
    assert torch.equal(x, x_orig)
    assert torch.equal(residual, residual_orig)

    sum_ref = x_orig + residual_orig
    assert torch.allclose(summed, sum_ref, atol=1e-6)
    norm_ref = _cpu_reference(sum_ref, norm.weight, norm.eps)
    assert torch.allclose(normalized, norm_ref, atol=1e-6)


# ================================================================ #7
def test_npu_rmsnorm_forward_returns_first_of_tuple(fresh_norm, monkeypatch):
    sentinel_out = torch.randn(2, 8)
    sentinel_rstd = torch.randn(2, 1)
    call_log = []

    def fake_rms_norm(x, weight, eps):
        call_log.append(("rms", x, weight, eps))
        return sentinel_out, sentinel_rstd

    _install_fake_torch_npu(monkeypatch, rms_norm_impl=fake_rms_norm)

    norm = fresh_norm.RMSNorm(8, 3e-6)
    norm.weight = torch.randn(8)

    fake_x = _FakeTensor(torch.randn(2, 8), "npu")
    y = norm.forward(fake_x)

    assert len(call_log) == 1
    assert call_log[0][0] == "rms"
    assert call_log[0][1] is fake_x
    assert call_log[0][2] is norm.weight
    assert call_log[0][3] == 3e-6
    assert y is sentinel_out  # tuple[1] discarded


# ================================================================ #8
def test_npu_rmsnorm_forward_inplace_calls_copy_(fresh_norm, monkeypatch):
    sentinel_out = torch.randn(3, 16)
    sentinel_rstd = torch.randn(3, 1)

    def fake_rms_norm(x, weight, eps):
        return sentinel_out, sentinel_rstd

    _install_fake_torch_npu(monkeypatch, rms_norm_impl=fake_rms_norm)

    norm = fresh_norm.RMSNorm(16, 1e-6)
    norm.weight = torch.randn(16)

    fake_x = _FakeTensor(torch.randn(3, 16), "npu")
    ret = norm.forward_inplace(fake_x)

    assert ret is None
    assert len(fake_x.copy_calls) == 1
    assert fake_x.copy_calls[0] is sentinel_out


# ================================================================ #9
def test_npu_fused_maps_first_and_third(fresh_norm, monkeypatch):
    normalized = torch.randn(4, 32)
    rstd = torch.randn(4, 1)
    summed = torch.randn(4, 32)
    call_log = []

    def fake_add_rms_norm(x, residual, weight, eps):
        call_log.append((x, residual, weight, eps))
        return normalized, rstd, summed

    _install_fake_torch_npu(monkeypatch, add_rms_norm_impl=fake_add_rms_norm)

    norm = fresh_norm.RMSNormFused(32, 5e-6)
    norm.weight = torch.randn(32)

    fake_x = _FakeTensor(torch.randn(4, 32), "npu")
    fake_residual = _FakeTensor(torch.randn(4, 32), "npu")

    out0, out1 = norm.forward(fake_x, fake_residual)

    assert len(call_log) == 1
    x_arg, res_arg, w_arg, eps_arg = call_log[0]
    assert x_arg is fake_x
    assert res_arg is fake_residual
    assert w_arg is norm.weight
    assert eps_arg == 5e-6
    assert out0 is normalized       # tuple[0]
    assert out1 is summed           # tuple[2] — rstd (tuple[1]) dropped


# ================================================================ #10
def test_npu_fused_does_not_mutate_inputs(fresh_norm, monkeypatch):
    normalized = torch.randn(3, 16)
    rstd = torch.randn(3, 1)
    summed = torch.randn(3, 16)

    def fake_add_rms_norm(x, residual, weight, eps):
        # Explicitly do NOT touch x or residual.
        return normalized, rstd, summed

    _install_fake_torch_npu(monkeypatch, add_rms_norm_impl=fake_add_rms_norm)

    norm = fresh_norm.RMSNormFused(16, 1e-6)
    norm.weight = torch.randn(16)

    real_x = torch.randn(3, 16)
    real_residual = torch.randn(3, 16)
    real_x_orig = real_x.clone()
    real_residual_orig = real_residual.clone()

    fake_x = _FakeTensor(real_x, "npu")
    fake_residual = _FakeTensor(real_residual, "npu")
    _ = norm.forward(fake_x, fake_residual)

    assert torch.equal(real_x, real_x_orig)
    assert torch.equal(real_residual, real_residual_orig)
    assert fake_x.copy_calls == []
    assert fake_residual.copy_calls == []


# ================================================================ #11
def test_cuda_paths_preserve_flashinfer_contracts(fresh_norm, monkeypatch):
    rmsnorm_calls = []
    fused_calls = []
    sentinel_new = torch.randn(3, 8)

    def fake_rmsnorm(x, weight, eps, out=None):
        rmsnorm_calls.append({"x": x, "weight": weight, "eps": eps, "out": out})
        return out if out is not None else sentinel_new

    def fake_fused_add_rmsnorm(x, residual, weight, eps):
        fused_calls.append({"x": x, "residual": residual, "weight": weight, "eps": eps})
        return None

    _install_fake_flashinfer(monkeypatch, fake_rmsnorm, fake_fused_add_rmsnorm)

    norm = fresh_norm.RMSNorm(8, 1e-6)
    norm.weight = torch.randn(8)

    # RMSNorm.forward — no out= kwarg
    fake_x = _FakeTensor(torch.randn(3, 8), "cuda")
    y = norm.forward(fake_x)
    assert y is sentinel_new
    assert rmsnorm_calls[-1]["out"] is None
    assert rmsnorm_calls[-1]["x"] is fake_x
    assert rmsnorm_calls[-1]["weight"] is norm.weight

    # RMSNorm.forward_inplace — out=x passthrough
    fake_x2 = _FakeTensor(torch.randn(3, 8), "cuda")
    ret = norm.forward_inplace(fake_x2)
    assert ret is None
    assert rmsnorm_calls[-1]["out"] is fake_x2
    assert rmsnorm_calls[-1]["x"] is fake_x2

    # RMSNormFused with residual — fused_add_rmsnorm invocation + identity return
    fused_norm = fresh_norm.RMSNormFused(8, 2e-6)
    fused_norm.weight = torch.randn(8)
    fake_x3 = _FakeTensor(torch.randn(3, 8), "cuda")
    fake_r = _FakeTensor(torch.randn(3, 8), "cuda")
    out0, out1 = fused_norm.forward(fake_x3, fake_r)
    assert out0 is fake_x3
    assert out1 is fake_r
    assert fused_calls[-1]["x"] is fake_x3
    assert fused_calls[-1]["residual"] is fake_r
    assert fused_calls[-1]["weight"] is fused_norm.weight
    assert fused_calls[-1]["eps"] == 2e-6

    # RMSNormFused residual=None — passthrough of x as second slot
    fake_x4 = _FakeTensor(torch.randn(3, 8), "cuda")
    o0, o1 = fused_norm.forward(fake_x4, None)
    assert o1 is fake_x4


# ================================================================ #12
def test_weight_attribute_name_and_shape_preserved(fresh_norm):
    norm1 = fresh_norm.RMSNorm(128, 1e-6)
    norm2 = fresh_norm.RMSNormFused(64, 1e-6)

    assert hasattr(norm1, "weight")
    assert hasattr(norm2, "weight")
    assert isinstance(norm1.weight, torch.Tensor)
    assert isinstance(norm2.weight, torch.Tensor)
    assert norm1.weight.shape == (128,)
    assert norm2.weight.shape == (64,)
    assert norm1.eps == 1e-6
    assert norm2.eps == 1e-6

    # BaseOP.state_dict serialization still surfaces exactly one "weight" key.
    assert list(norm1.state_dict().keys()) == ["weight"]
    assert list(norm2.state_dict().keys()) == ["weight"]


# ================================================================ #13
def test_missing_torch_npu_raises_runtime_error(fresh_norm, block_torch_npu):
    norm = fresh_norm.RMSNorm(8, 1e-6)
    norm.weight = torch.randn(8)
    fake_x = _FakeTensor(torch.randn(1, 8), "npu")
    with pytest.raises(RuntimeError, match="torch_npu"):
        norm.forward(fake_x)


def test_missing_flashinfer_raises_runtime_error(fresh_norm, block_flashinfer):
    norm = fresh_norm.RMSNorm(8, 1e-6)
    norm.weight = torch.randn(8)
    fake_x = _FakeTensor(torch.randn(1, 8), "cuda")
    with pytest.raises(RuntimeError, match="flashinfer"):
        norm.forward(fake_x)


def test_missing_flashinfer_fused_add_raises_runtime_error(fresh_norm, block_flashinfer):
    norm = fresh_norm.RMSNormFused(8, 1e-6)
    norm.weight = torch.randn(8)
    fake_x = _FakeTensor(torch.randn(1, 8), "cuda")
    fake_r = _FakeTensor(torch.randn(1, 8), "cuda")
    with pytest.raises(RuntimeError, match="flashinfer"):
        norm.forward(fake_x, fake_r)
