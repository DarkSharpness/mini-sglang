"""Gate 1.9o hermetic tests for silu_and_mul CUDA/NPU/CPU dispatch.

Neither real CUDA nor real NPU are required. The CUDA and NPU branches are
exercised via fake device tensors and fake modules injected into sys.modules
so we can assert dispatch behavior deterministically on a CPU host.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = "minisgl.layers.activation"
SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "minisgl"
    / "layers"
    / "activation.py"
)


@pytest.fixture()
def activation_mod():
    mod = importlib.import_module(MODULE_PATH)
    return mod


# =================================================================== helpers
def _cpu_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


class _FakeDevice:
    """Fake device that looks like an NPU or CUDA tensor's .device."""

    def __init__(self, type_: str, index: int = 0):
        self.type = type_
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, _FakeDevice)
            and self.type == other.type
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):
        return f"_FakeDevice(type={self.type!r}, index={self.index})"


class _FakeTensor:
    """Minimal tensor stand-in exposing the attributes silu_and_mul touches."""

    def __init__(
        self,
        shape,
        dtype=torch.float16,
        device_type: str = "npu",
        device_index: int = 0,
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = _FakeDevice(device_type, device_index)
        # side-effect counters
        self.contiguous_calls = 0
        self.float_calls = 0
        self.long_calls = 0
        self.chunk_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self

    def float(self):
        self.float_calls += 1
        return self

    def long(self):
        self.long_calls += 1
        return self

    def chunk(self, n, dim=-1):
        self.chunk_calls += 1
        raise AssertionError("chunk should not be called on NPU/CUDA dispatch branches")


# ================================================================== 1. import
def test_import_does_not_pull_flashinfer_or_torch_npu():
    for name in list(sys.modules):
        if name == MODULE_PATH:
            del sys.modules[name]
    for name in ("flashinfer", "torch_npu"):
        sys.modules.pop(name, None)

    importlib.import_module(MODULE_PATH)

    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules


# =============================================================== 2/3/4/5 CPU
def test_cpu_contiguous_matches_reference(activation_mod):
    torch.manual_seed(1701)
    x = torch.randn(4, 8, dtype=torch.float32)
    y = activation_mod.silu_and_mul(x)
    assert torch.equal(y, _cpu_reference(x))


def test_cpu_non_contiguous_matches_reference(activation_mod):
    torch.manual_seed(1702)
    packed = torch.randn(4, 16, dtype=torch.float32)
    x = packed[:, 4:12]
    assert not x.is_contiguous()
    y = activation_mod.silu_and_mul(x)
    assert torch.equal(y, _cpu_reference(x))


def test_cpu_input_not_mutated(activation_mod):
    torch.manual_seed(1703)
    x = torch.randn(3, 6, dtype=torch.float32)
    snap = x.detach().clone()
    ptr = x.data_ptr()
    _ = activation_mod.silu_and_mul(x)
    assert torch.equal(x, snap)
    assert x.data_ptr() == ptr


def test_cpu_returns_new_tensor(activation_mod):
    torch.manual_seed(1704)
    x = torch.randn(2, 4, dtype=torch.float32)
    y = activation_mod.silu_and_mul(x)
    assert y.data_ptr() != x.data_ptr()


# ================================================================== 6. CPU out
def test_cpu_out_returns_same_object_and_correct_values(activation_mod):
    torch.manual_seed(1705)
    x = torch.randn(3, 8, dtype=torch.float32)
    out = torch.empty(3, 4, dtype=torch.float32)
    out_ptr = out.data_ptr()

    result = activation_mod.silu_and_mul(x, out=out)

    assert result is out
    assert result.data_ptr() == out_ptr
    assert torch.equal(result, _cpu_reference(x))


def test_cpu_out_none_returns_new_tensor(activation_mod):
    torch.manual_seed(1706)
    x = torch.randn(2, 4, dtype=torch.float32)
    y = activation_mod.silu_and_mul(x, out=None)
    assert torch.equal(y, _cpu_reference(x))
    assert y.data_ptr() != x.data_ptr()


# ==================================================================== 7-10 NPU
def _install_fake_torch_npu(swiglu_impl):
    fake = ModuleType("torch_npu")
    fake.npu_swiglu = swiglu_impl
    sys.modules["torch_npu"] = fake
    return fake


def test_fake_npu_calls_npu_swiglu_once_with_dim_neg1(activation_mod):
    calls = []

    def _swiglu(x, dim=-999):
        calls.append({"x": x, "dim": dim, "kw_dim_used": dim})
        return torch.zeros(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype)

    x = _FakeTensor((3, 8), dtype=torch.float16, device_type="npu")

    real_zeros = torch.zeros

    def _swiglu_returning_tensor(x_, dim=-999):
        calls.append({"dim": dim})
        return real_zeros(3, 4, dtype=torch.float16)

    try:
        _install_fake_torch_npu(_swiglu_returning_tensor)
        y = activation_mod.silu_and_mul(x)
    finally:
        sys.modules.pop("torch_npu", None)

    assert len(calls) == 1
    assert calls[0]["dim"] == -1
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (3, 4)


def test_fake_npu_does_not_call_contiguous_or_cast(activation_mod):
    x = _FakeTensor((2, 6), dtype=torch.float16, device_type="npu")

    def _swiglu(x_, dim=-1):
        # x_ is the same FakeTensor — mutating internal counters proves identity
        assert x_ is x
        return torch.zeros(2, 3, dtype=torch.float16)

    try:
        _install_fake_torch_npu(_swiglu)
        activation_mod.silu_and_mul(x)
    finally:
        sys.modules.pop("torch_npu", None)

    assert x.contiguous_calls == 0
    assert x.float_calls == 0
    assert x.long_calls == 0
    assert x.chunk_calls == 0


def test_fake_npu_input_not_mutated(activation_mod):
    # Use a real CPU tensor spoofed as NPU via mock on .device
    real = torch.randn(2, 4, dtype=torch.float16)
    snap = real.detach().clone()

    class _Spoofed:
        def __init__(self, t):
            self._t = t
            self.shape = t.shape
            self.dtype = t.dtype
            self.device = _FakeDevice("npu", 0)

    spoofed = _Spoofed(real)

    def _swiglu(x_, dim=-1):
        # simulate NPU op that does not modify input
        return torch.empty(2, 2, dtype=torch.float16)

    try:
        _install_fake_torch_npu(_swiglu)
        activation_mod.silu_and_mul(spoofed)
    finally:
        sys.modules.pop("torch_npu", None)

    assert torch.equal(real, snap)


def test_fake_npu_out_semantics(activation_mod):
    x = _FakeTensor((2, 4), dtype=torch.float16, device_type="npu")

    filled = torch.arange(4, dtype=torch.float16).reshape(2, 2)

    class _OutSpy:
        def __init__(self):
            self.shape = torch.Size((2, 2))
            self.dtype = torch.float16
            self.device = _FakeDevice("npu", 0)
            self.copied_from = None

        def copy_(self, src):
            self.copied_from = src
            return self

    out = _OutSpy()

    def _swiglu(x_, dim=-1):
        return filled

    try:
        _install_fake_torch_npu(_swiglu)
        result = activation_mod.silu_and_mul(x, out=out)
    finally:
        sys.modules.pop("torch_npu", None)

    assert result is out
    assert out.copied_from is filled


# ==================================================================== 11 CUDA
def test_fake_cuda_preserves_flashinfer_args_and_return(activation_mod):
    seen = {}

    def _fake_flashinfer_silu_and_mul(x, out=None):
        seen["x"] = x
        seen["out"] = out
        seen["called"] = seen.get("called", 0) + 1
        return "SENTINEL_RETURN"

    fake_fi = ModuleType("flashinfer")
    fake_fi.silu_and_mul = _fake_flashinfer_silu_and_mul

    class _CudaTensor:
        def __init__(self, shape, dtype=torch.float16):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.device = _FakeDevice("cuda", 0)

    x = _CudaTensor((2, 8))
    out_obj = _CudaTensor((2, 4))

    try:
        sys.modules["flashinfer"] = fake_fi
        r_none = activation_mod.silu_and_mul(x, out=None)
        r_out = activation_mod.silu_and_mul(x, out=out_obj)
    finally:
        sys.modules.pop("flashinfer", None)

    assert r_none == "SENTINEL_RETURN"
    assert r_out == "SENTINEL_RETURN"
    assert seen["called"] == 2
    assert seen["x"] is x
    assert seen["out"] is out_obj


# ============================================================ 12 missing deps
def test_missing_torch_npu_raises_clear_runtimeerror(activation_mod):
    x = _FakeTensor((2, 4), dtype=torch.float16, device_type="npu")

    # Prevent any real torch_npu from resolving
    sys.modules.pop("torch_npu", None)
    with mock.patch.dict(sys.modules, {"torch_npu": None}):
        with pytest.raises(RuntimeError, match="torch_npu"):
            activation_mod.silu_and_mul(x)


def test_missing_flashinfer_raises_clear_runtimeerror(activation_mod):
    class _CudaTensor:
        shape = torch.Size((2, 4))
        dtype = torch.float16
        device = _FakeDevice("cuda", 0)

    sys.modules.pop("flashinfer", None)
    with mock.patch.dict(sys.modules, {"flashinfer": None}):
        with pytest.raises(RuntimeError, match="flashinfer"):
            activation_mod.silu_and_mul(_CudaTensor())


# ============================================================ 13 odd last dim
def test_odd_last_dim_rejected(activation_mod):
    x = torch.randn(2, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="even last dimension"):
        activation_mod.silu_and_mul(x)


# ============================================================ 14 out mismatches
def test_out_shape_mismatch_rejected(activation_mod):
    x = torch.randn(2, 8, dtype=torch.float32)
    bad = torch.empty(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        activation_mod.silu_and_mul(x, out=bad)


def test_out_dtype_mismatch_rejected(activation_mod):
    x = torch.randn(2, 8, dtype=torch.float32)
    bad = torch.empty(2, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        activation_mod.silu_and_mul(x, out=bad)


def test_out_device_mismatch_rejected(activation_mod):
    # Use FakeTensor to spoof device without needing a second real device.
    class _RealShaped:
        def __init__(self):
            self.shape = torch.Size((2, 8))
            self.dtype = torch.float32
            self.device = _FakeDevice("cpu", 0)

    class _WrongDev:
        def __init__(self):
            self.shape = torch.Size((2, 4))
            self.dtype = torch.float32
            self.device = _FakeDevice("cuda", 0)

    with pytest.raises(ValueError, match="device"):
        activation_mod.silu_and_mul(_RealShaped(), out=_WrongDev())


# ============================================================ 15 gelu unchanged
def test_gelu_and_mul_source_unchanged():
    src = SOURCE_PATH.read_text()
    tree = ast.parse(src)
    gelu_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "gelu_and_mul"
    ]
    assert len(gelu_defs) == 1
    fn = gelu_defs[0]
    # Body must be: from flashinfer import gelu_and_mul ; return gelu_and_mul(x, out=out)
    assert len(fn.body) == 2
    stmt0 = fn.body[0]
    assert isinstance(stmt0, ast.ImportFrom)
    assert stmt0.module == "flashinfer"
    assert [a.name for a in stmt0.names] == ["gelu_and_mul"]
    stmt1 = fn.body[1]
    assert isinstance(stmt1, ast.Return)
    assert isinstance(stmt1.value, ast.Call)
    assert isinstance(stmt1.value.func, ast.Name)
    assert stmt1.value.func.id == "gelu_and_mul"


def test_gelu_and_mul_still_delegates_to_flashinfer(activation_mod):
    seen = {}

    def _fake_gelu(x, out=None):
        seen["x"] = x
        seen["out"] = out
        return "GELU_SENTINEL"

    fake = ModuleType("flashinfer")
    fake.gelu_and_mul = _fake_gelu

    sentinel_x = object()
    sentinel_out = object()

    try:
        sys.modules["flashinfer"] = fake
        result = activation_mod.gelu_and_mul(sentinel_x, out=sentinel_out)
    finally:
        sys.modules.pop("flashinfer", None)

    assert result == "GELU_SENTINEL"
    assert seen["x"] is sentinel_x
    assert seen["out"] is sentinel_out
