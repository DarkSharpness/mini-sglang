"""Hermetic tests for ``python/minisgl/layers/rotary.py`` device dispatch.

The tests never touch CUDA or NPU hardware. Fake ``flashinfer`` and
``torch_npu`` modules are injected via ``monkeypatch`` and a fake tensor
wrapper carries a controlled ``device.type`` string through the dispatch
while forwarding ``.dtype`` / ``.shape`` / ``.view`` to a real underlying
tensor.
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
def fresh_rotary(clean_optional_deps):
    """Re-import ``minisgl.layers.rotary`` on a clean slate."""
    if "minisgl.layers.rotary" in sys.modules:
        del sys.modules["minisgl.layers.rotary"]
    import minisgl.layers.rotary as rotary
    return rotary


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


# ---------- npu device shim (hermetic, no torch_npu required) --------------
#
# ``torch.device('npu[...]')`` string parsing fails on hosts without
# ``torch_npu`` — the parser doesn't know the ``npu`` type name. Tests that
# only need to *reason about* an ``npu`` device (cache-key isolation,
# accelerator-index rejection, etc.) shouldn't require torch_npu wheels.
#
# This fixture intercepts ``torch.device('npu', ...)`` / ``torch.device('npu:i')``
# with a lightweight stand-in that mimics the surface ``_normalize_device``
# consumes: ``.type``, ``.index``, equality, and hashability. All other
# ``torch.device(...)`` inputs (cpu, meta, cuda:i) fall through to the real
# parser.


class _NpuDeviceStub:
    __slots__ = ("type", "index")

    def __init__(self, index):
        self.type = "npu"
        self.index = index

    def __repr__(self):
        if self.index is None:
            return "device(type='npu')"
        return f"device(type='npu', index={self.index})"

    def __eq__(self, other):
        return (
            isinstance(other, _NpuDeviceStub)
            and self.type == other.type
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.type, self.index))


@pytest.fixture
def npu_device_shim(monkeypatch):
    """Return a ``make(type, index=None)`` builder that produces a stand-in
    ``torch.device('npu[:i]')`` while patching ``torch.device`` so
    ``_normalize_device``'s own ``torch.device(device)`` re-wrap step accepts
    the stub. Non-npu calls fall through to the real constructor unchanged.
    """
    real_torch_device = torch.device

    def fake_device(*args, **kwargs):
        # torch.device(stub) → identity for our stub
        if len(args) == 1 and isinstance(args[0], _NpuDeviceStub):
            return args[0]
        # torch.device("npu", idx)
        if len(args) >= 1 and args[0] == "npu":
            idx = args[1] if len(args) > 1 else kwargs.get("index")
            return _NpuDeviceStub(idx)
        # torch.device("npu:i") string form
        if len(args) == 1 and isinstance(args[0], str) and args[0].startswith("npu"):
            s = args[0]
            if s == "npu":
                return _NpuDeviceStub(None)
            if ":" in s:
                _, idx_s = s.split(":", 1)
                return _NpuDeviceStub(int(idx_s))
        return real_torch_device(*args, **kwargs)

    monkeypatch.setattr(torch, "device", fake_device)

    def make(_type, index=None):
        assert _type == "npu", "npu_device_shim only synthesises 'npu' devices"
        return _NpuDeviceStub(index)

    return make


# ============================================================ helpers
class _FakeDevice:
    def __init__(self, t: str) -> None:
        self.type = t

    def __repr__(self) -> str:
        return f"_FakeDevice(type={self.type!r})"


class _FakeTensor:
    """Minimal tensor stand-in.

    The dispatch code reads ``.device.type`` for branching, ``.dtype`` for
    the cos/sin cast, ``.shape`` for reshaping, and calls ``.view(...)`` to
    build the 4-D BSND input for ``npu_rotary_mul``. The FlashInfer path
    passes ``self`` through unchanged as the return value.
    """

    def __init__(self, real: torch.Tensor, device_type: str) -> None:
        self._real = real
        self.device = _FakeDevice(device_type)

    @property
    def dtype(self):
        return self._real.dtype

    @property
    def shape(self):
        return self._real.shape

    def view(self, *args):
        return self._real.view(*args)


def _cpu_reference_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone NeoX rotate_half reference, computed in fp32.

    Indexes ``cos_sin_cache`` with ``positions`` directly — no dtype
    coercion, mirroring the production dispatch policy.
    """
    half = rotary_dim // 2
    selected = cos_sin_cache[positions]
    cos_half = selected[..., :half]
    sin_half = selected[..., half:]
    cos_full = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)  # (T, 1, D)
    sin_full = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x1 = x_f[..., :half]
        x2 = x_f[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_f * cos_full + rotated * sin_full).to(x.dtype)

    return _rotate(query), _rotate(key)


class _StrictPositions(torch.Tensor):
    """torch.Tensor subclass whose dtype-coercion methods raise.

    Any call to ``.to()``, ``.long()`` or ``.type()`` on an instance fails
    the test loudly — the RoPE dispatch must accept the caller's positions
    dtype (int32 or int64) verbatim and index ``_cos_sin_cache`` directly.

    ``__torch_function__`` unwraps ``_StrictPositions`` args to plain
    ``torch.Tensor`` before delegating to the real op, so downstream results
    (``cache[positions]`` → ``cos_full`` → …) are plain tensors and don't
    inherit the raising overrides. Only direct method calls on a positions
    instance can trip the guard.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Second line of defence: catch ``.to`` / ``.long`` / ``.type`` calls
        # that arrive here rather than via Python-level method resolution.
        fname = getattr(func, "__name__", "")
        if fname in {"to", "long", "type"}:
            for a in args:
                if isinstance(a, _StrictPositions):
                    raise AssertionError(
                        f"positions.{fname}(...) reached via __torch_function__ "
                        "— RoPE dispatch must not coerce positions dtype"
                    )

        # Disable subclass dispatch for the duration of unwrap + delegated
        # call, so ``_make_subclass`` and the real op don't re-enter here.
        with torch._C.DisableTorchFunctionSubclass():
            def _unwrap(x):
                if isinstance(x, _StrictPositions):
                    return torch.Tensor._make_subclass(torch.Tensor, x)
                return x

            new_args = tuple(_unwrap(a) for a in args)
            new_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)

    def to(self, *args, **kwargs):
        raise AssertionError(
            f"positions.to({args!r}, {kwargs!r}) called — RoPE dispatch must "
            "not coerce positions dtype"
        )

    def long(self, *args, **kwargs):
        raise AssertionError(
            "positions.long() called — RoPE dispatch must not coerce positions"
        )

    def type(self, *args, **kwargs):
        raise AssertionError(
            f"positions.type({args!r}) called — RoPE dispatch must not coerce"
        )


def _make_strict(values_or_tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(values_or_tensor, torch.Tensor):
        base = values_or_tensor
    else:
        base = torch.tensor(values_or_tensor, dtype=dtype)
    # ``_make_subclass`` is a C-level factory that doesn't dispatch through
    # ``__torch_function__``, avoiding infinite recursion during setup.
    return torch.Tensor._make_subclass(_StrictPositions, base)


def _install_fake_flashinfer(monkeypatch, apply_rope=None):
    fake = types.ModuleType("flashinfer")
    if apply_rope is not None:
        fake.apply_rope_with_cos_sin_cache_inplace = apply_rope
    monkeypatch.setitem(sys.modules, "flashinfer", fake)


def _install_fake_torch_npu(monkeypatch, rotary_mul=None):
    fake = types.ModuleType("torch_npu")
    if rotary_mul is not None:
        fake.npu_rotary_mul = rotary_mul
    monkeypatch.setitem(sys.modules, "torch_npu", fake)


# ================================================================ #1
def test_import_rotary_does_not_trigger_optional_deps(fresh_rotary):
    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules


# ================================================================ #2
def test_construct_rotary_does_not_trigger_optional_deps(fresh_rotary):
    _ = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    assert "flashinfer" not in sys.modules
    assert "torch_npu" not in sys.modules


# ================================================================ #3
def test_construct_preserves_params_and_cache_shape(fresh_rotary):
    rope = fresh_rotary.RotaryEmbedding(128, 128, 64, 10000.0)
    assert rope.head_size == 128
    assert rope.rotary_dim == 128
    assert rope._cos_sin_cache.shape == (64, 128)
    assert rope._cos_sin_cache.dtype == torch.float32
    # StateLessOP short-circuits state_dict to empty — cache must not leak.
    assert rope.state_dict() == {}


# ================================================================ #4
def test_cpu_forward_matches_neox_reference_fp32(fresh_rotary):
    torch.manual_seed(4001)
    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 6, 4, 2, 128
    positions = torch.tensor([0, 1, 2, 5, 7, 9], dtype=torch.int64)
    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)

    q_out, k_out = rope.forward(positions, query, key)

    q_ref, k_ref = _cpu_reference_rope(query, key, positions, rope._cos_sin_cache, 128)
    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    assert q_out.dtype == query.dtype
    assert k_out.dtype == key.dtype
    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)


# ================================================================ #5
def test_cpu_forward_returns_new_tensors_and_inputs_untouched(fresh_rotary):
    torch.manual_seed(4002)
    rope = fresh_rotary.RotaryEmbedding(64, 64, 32, 10000.0)
    T, Hq, Hk, D = 4, 3, 1, 64
    positions = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)
    q_orig = query.clone()
    k_orig = key.clone()
    q_ptr, k_ptr = query.data_ptr(), key.data_ptr()

    q_out, k_out = rope.forward(positions, query, key)

    assert torch.equal(query, q_orig)
    assert torch.equal(key, k_orig)
    assert q_out.data_ptr() != q_ptr
    assert k_out.data_ptr() != k_ptr


# ================================================================ #6
def test_cpu_forward_accepts_non_contiguous_positions(fresh_rotary):
    torch.manual_seed(4003)
    rope = fresh_rotary.RotaryEmbedding(64, 64, 32, 10000.0)
    T, Hq, Hk, D = 4, 2, 2, 64
    # Non-contiguous slice — stride 2 view over an 8-element vector, wrapped
    # as a _StrictPositions so any .to()/.long()/.type() from the dispatch
    # would raise immediately.
    base = torch.tensor([0, 99, 1, 99, 3, 99, 5, 99], dtype=torch.int64)
    positions = _make_strict(base[::2])
    assert not positions.is_contiguous()
    assert positions.shape == (4,)
    pos_id_pre = id(positions)
    pos_dtype_pre = positions.dtype

    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)

    q_out, k_out = rope.forward(positions, query, key)

    # Same object, same dtype after the call.
    assert id(positions) == pos_id_pre
    assert positions.dtype == pos_dtype_pre == torch.int64

    ref_positions = torch.tensor([0, 1, 3, 5], dtype=torch.int64)
    q_ref, k_ref = _cpu_reference_rope(query, key, ref_positions, rope._cos_sin_cache, 64)
    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)


# ================================================================ #7
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_cpu_forward_accepts_int32_and_int64_positions(fresh_rotary, pos_dtype):
    torch.manual_seed(4004)
    rope = fresh_rotary.RotaryEmbedding(64, 64, 32, 10000.0)
    T, Hq, Hk, D = 5, 2, 1, 64
    values = [0, 1, 2, 4, 8]
    positions = _make_strict(values, pos_dtype)
    pos_id_pre = id(positions)
    pos_dtype_pre = positions.dtype
    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)

    q_out, k_out = rope.forward(positions, query, key)

    # positions object identity + dtype preserved through the dispatch.
    assert id(positions) == pos_id_pre
    assert positions.dtype == pos_dtype_pre == pos_dtype

    # Numerical correctness — compared against a plain-int64 reference.
    ref_positions = torch.tensor(values, dtype=torch.int64)
    q_ref, k_ref = _cpu_reference_rope(query, key, ref_positions, rope._cos_sin_cache, 64)
    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)


# ================================================================ #7b
def test_int32_and_int64_positions_produce_identical_outputs(fresh_rotary):
    """Same position indices in int32 vs int64 must yield bit-identical output."""
    torch.manual_seed(4009)
    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 4, 2, 1, 128
    values = [0, 2, 5, 8]
    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)
    q_clone = query.clone()
    k_clone = key.clone()

    positions_i32 = _make_strict(values, torch.int32)
    q_i32, k_i32 = rope.forward(positions_i32, query, key)
    assert positions_i32.dtype == torch.int32   # unchanged

    positions_i64 = _make_strict(values, torch.int64)
    q_i64, k_i64 = rope.forward(positions_i64, q_clone, k_clone)
    assert positions_i64.dtype == torch.int64   # unchanged

    assert torch.equal(q_i32, q_i64)
    assert torch.equal(k_i32, k_i64)


# ================================================================ #8
def test_cpu_forward_supports_different_hq_hk_head_counts(fresh_rotary):
    torch.manual_seed(4005)
    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, D = 3, 128
    Hq, Hk = 8, 1   # extreme GQA — 8 query heads, 1 KV head.
    positions = torch.tensor([2, 4, 6], dtype=torch.int64)
    query = torch.randn(T, Hq, D, dtype=torch.float32)
    key = torch.randn(T, Hk, D, dtype=torch.float32)

    q_out, k_out = rope.forward(positions, query, key)
    assert q_out.shape == (T, Hq, D)
    assert k_out.shape == (T, Hk, D)
    q_ref, k_ref = _cpu_reference_rope(query, key, positions, rope._cos_sin_cache, 128)
    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)


# ================================================================ #9
def test_npu_forward_reshapes_qk_to_4d_bsnd(fresh_rotary, monkeypatch):
    call_log = []

    def fake_rotary_mul(x, r1, r2, rotary_mode):
        call_log.append({"x_shape": tuple(x.shape), "r1_shape": tuple(r1.shape),
                         "r2_shape": tuple(r2.shape), "rotary_mode": rotary_mode})
        return torch.zeros_like(x)

    _install_fake_torch_npu(monkeypatch, rotary_mul=fake_rotary_mul)

    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 5, 4, 2, 128
    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    fake_q = _FakeTensor(torch.randn(T, Hq, D, dtype=torch.float16), "npu")
    fake_k = _FakeTensor(torch.randn(T, Hk, D, dtype=torch.float16), "npu")

    _ = rope.forward(positions, fake_q, fake_k)

    assert len(call_log) == 2
    # Query call
    assert call_log[0]["x_shape"] == (1, T, Hq, D)
    # Key call
    assert call_log[1]["x_shape"] == (1, T, Hk, D)


# ================================================================ #10
def test_npu_forward_cos_sin_shape_is_1_T_1_D_neox_split(fresh_rotary, monkeypatch):
    captured = []

    def fake_rotary_mul(x, r1, r2, rotary_mode):
        captured.append({"r1": r1.clone(), "r2": r2.clone()})
        return torch.zeros_like(x)

    _install_fake_torch_npu(monkeypatch, rotary_mul=fake_rotary_mul)

    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 3, 2, 1, 128
    positions = torch.tensor([1, 2, 5], dtype=torch.int64)
    fake_q = _FakeTensor(torch.randn(T, Hq, D, dtype=torch.float16), "npu")
    fake_k = _FakeTensor(torch.randn(T, Hk, D, dtype=torch.float16), "npu")

    _ = rope.forward(positions, fake_q, fake_k)

    assert len(captured) == 2
    for entry in captured:
        assert entry["r1"].shape == (1, T, 1, D)
        assert entry["r2"].shape == (1, T, 1, D)
        assert entry["r1"].dtype == torch.float16
        assert entry["r2"].dtype == torch.float16

    # NeoX split correctness: cos_full == cat(cos_half, cos_half); the two
    # halves of the last dim must be equal to each other.
    r1 = captured[0]["r1"]
    half = D // 2
    assert torch.equal(r1[..., :half], r1[..., half:])
    r2 = captured[0]["r2"]
    assert torch.equal(r2[..., :half], r2[..., half:])

    # And the halves must match what the shared cache actually stores.
    selected = rope._cos_sin_cache[positions.long()]        # (T, D) fp32
    cos_half_ref = selected[..., :half].to(torch.float16)   # (T, half)
    sin_half_ref = selected[..., half:].to(torch.float16)
    # r1 has shape (1, T, 1, D) — squeeze to (T, D) then take first half.
    assert torch.equal(r1.view(T, D)[..., :half], cos_half_ref)
    assert torch.equal(r2.view(T, D)[..., :half], sin_half_ref)


# ================================================================ #11
def test_npu_forward_calls_npu_rotary_mul_twice_with_rotary_mode_half(fresh_rotary, monkeypatch):
    call_log = []

    def fake_rotary_mul(x, r1, r2, rotary_mode):
        call_log.append(rotary_mode)
        return torch.zeros_like(x)

    _install_fake_torch_npu(monkeypatch, rotary_mul=fake_rotary_mul)

    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 4, 3, 3, 128
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    fake_q = _FakeTensor(torch.randn(T, Hq, D, dtype=torch.float16), "npu")
    fake_k = _FakeTensor(torch.randn(T, Hk, D, dtype=torch.float16), "npu")

    _ = rope.forward(positions, fake_q, fake_k)

    assert call_log == ["half", "half"]


# ================================================================ #12
def test_npu_forward_returns_new_tensors_no_copy_no_contiguous(fresh_rotary, monkeypatch):
    sentinel_q_4d = torch.randn(1, 4, 3, 128, dtype=torch.float16)
    sentinel_k_4d = torch.randn(1, 4, 1, 128, dtype=torch.float16)
    seen = []

    def fake_rotary_mul(x, r1, r2, rotary_mode):
        seen.append(tuple(x.shape))
        if x.shape[2] == 3:
            return sentinel_q_4d
        return sentinel_k_4d

    _install_fake_torch_npu(monkeypatch, rotary_mul=fake_rotary_mul)

    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    T, Hq, Hk, D = 4, 3, 1, 128
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    real_q = torch.randn(T, Hq, D, dtype=torch.float16)
    real_k = torch.randn(T, Hk, D, dtype=torch.float16)
    q_orig = real_q.clone()
    k_orig = real_k.clone()
    fake_q = _FakeTensor(real_q, "npu")
    fake_k = _FakeTensor(real_k, "npu")

    q_out, k_out = rope.forward(positions, fake_q, fake_k)

    # Inputs never mutated (no copy_).
    assert torch.equal(real_q, q_orig)
    assert torch.equal(real_k, k_orig)
    # Returned tensors are derived from the sentinels (no clone / contiguous
    # inserted — .squeeze(0) is a view, so data_ptr matches the sentinels).
    assert q_out.data_ptr() == sentinel_q_4d.data_ptr()
    assert k_out.data_ptr() == sentinel_k_4d.data_ptr()
    assert q_out.shape == (T, Hq, D)
    assert k_out.shape == (T, Hk, D)


# ================================================================ #13
def test_cuda_forward_delegates_to_flashinfer_inplace_returns_identity(
    fresh_rotary, monkeypatch,
):
    call_log = []

    def fake_apply_rope(positions, query, key, head_size, cos_sin_cache):
        call_log.append({
            "positions": positions, "query": query, "key": key,
            "head_size": head_size, "cos_sin_cache": cos_sin_cache,
        })
        return None

    _install_fake_flashinfer(monkeypatch, apply_rope=fake_apply_rope)

    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    fake_q = _FakeTensor(torch.randn(4, 8, 128), "cuda")
    fake_k = _FakeTensor(torch.randn(4, 2, 128), "cuda")

    q_out, k_out = rope.forward(positions, fake_q, fake_k)

    # Identity return — FlashInfer mutates in-place; caller expects the same
    # tensor objects back.
    assert q_out is fake_q
    assert k_out is fake_k
    assert len(call_log) == 1
    entry = call_log[0]
    assert entry["positions"] is positions
    assert entry["query"] is fake_q
    assert entry["key"] is fake_k
    assert entry["head_size"] == 128
    assert entry["cos_sin_cache"] is rope._cos_sin_cache


# ================================================================ #14
def test_missing_optional_deps_raise_runtime_error(
    fresh_rotary, block_flashinfer, block_torch_npu,
):
    rope = fresh_rotary.RotaryEmbedding(128, 128, 32, 10000.0)
    positions = torch.tensor([0, 1], dtype=torch.int64)

    fake_q_cuda = _FakeTensor(torch.randn(2, 2, 128), "cuda")
    fake_k_cuda = _FakeTensor(torch.randn(2, 1, 128), "cuda")
    with pytest.raises(RuntimeError, match="flashinfer"):
        rope.forward(positions, fake_q_cuda, fake_k_cuda)

    fake_q_npu = _FakeTensor(torch.randn(2, 2, 128, dtype=torch.float16), "npu")
    fake_k_npu = _FakeTensor(torch.randn(2, 1, 128, dtype=torch.float16), "npu")
    with pytest.raises(RuntimeError, match="torch_npu"):
        rope.forward(positions, fake_q_npu, fake_k_npu)


# ================================================================ #15
def test_rotary_source_never_coerces_positions_dtype(fresh_rotary):
    """AST guard: ``positions.to(...)`` / ``.long()`` / ``.type(...)`` must
    not appear anywhere in ``rotary.py`` — the dispatch has to index the
    cache with the caller's positions verbatim.
    """
    import ast
    import inspect

    src = inspect.getsource(fresh_rotary)
    tree = ast.parse(src)
    forbidden = {"to", "long", "type"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # Only flag ``positions.<name>(...)``.
        if isinstance(func.value, ast.Name) and func.value.id == "positions":
            if func.attr in forbidden:
                offenders.append(f"positions.{func.attr}(...) at line {node.lineno}")
    assert offenders == [], (
        "rotary.py must not coerce ``positions`` dtype; found: "
        + "; ".join(offenders)
    )


# ==================================================================== #16
# ---------- Device-keyed cache tests --------------------------------------
#
# These tests exercise the split between the public ``get_rope`` (which
# resolves a target device from setter / ambient) and the private
# ``_get_rope_cached`` (which is keyed on (params, device)).
#
# We deliberately never enter ``with torch.device('npu:0'):`` from these
# tests: on CI hosts without ``torch_npu`` that would raise from PyTorch's
# device dispatch. Instead we monkeypatch ``rotary._build_rope`` — the
# private constructor shim that owns the device context — with a stub that
# returns a fresh sentinel per call and records what it was asked to build.
# The cache-key logic under test lives entirely in ``_get_rope_cached`` and
# is orthogonal to whether real allocation happens.


class _RopeSentinel:
    """Lightweight stand-in for ``RotaryEmbedding`` returned by the stub."""

    __slots__ = ("device", "tag")

    def __init__(self, device: torch.device, tag: object) -> None:
        self.device = device
        self.tag = tag


@pytest.fixture
def stub_build(fresh_rotary):
    """Rebind ``rotary._build_rope`` to a stub that never touches PyTorch's
    device dispatch.

    ``_get_rope_cached`` resolves ``_build_rope`` from the module globals at
    call time, so simply reassigning the attribute is enough — no cache
    surgery required. We also clear any residual cache entries so tests
    that run after another test in the same module see a virgin cache.
    """
    calls: list[dict] = []

    def stub(head_dim, rotary_dim, max_position, base, rope_scaling, device):
        payload = dict(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            rope_scaling=rope_scaling,
            device=device,
        )
        calls.append(payload)
        return _RopeSentinel(device=device, tag=len(calls))

    fresh_rotary._build_rope = stub
    fresh_rotary.get_rope.cache_clear()
    return calls


# --- resolver rules -------------------------------------------------------

def test_resolve_setter_wins_when_no_meta(fresh_rotary):
    """(rule 1) Setter takes precedence over the ambient CPU default."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    resolved = fresh_rotary._resolve_rope_device()
    assert resolved.type == "cuda"
    assert resolved.index == 0


def test_resolve_setter_wins_under_meta(fresh_rotary):
    """(rule 1 under meta) Setter still wins inside ``torch.device('meta')``."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    with torch.device("meta"):
        resolved = fresh_rotary._resolve_rope_device()
    assert resolved.type == "cuda"
    assert resolved.index == 0


def test_resolve_falls_back_to_ambient_cpu(fresh_rotary):
    """(rule 2) With no setter and no meta scope, ambient CPU is returned."""
    assert fresh_rotary._ROPE_DEVICE is None
    resolved = fresh_rotary._resolve_rope_device()
    assert resolved == torch.device("cpu")


def test_resolve_raises_on_meta_without_setter(fresh_rotary):
    """(rule 3) Meta ambient with no setter must raise a clear ``RuntimeError``."""
    assert fresh_rotary._ROPE_DEVICE is None
    with torch.device("meta"):
        with pytest.raises(RuntimeError, match="set_rope_device"):
            fresh_rotary._resolve_rope_device()


def test_normalize_device_cpu_variants_collapse(fresh_rotary):
    """``cpu`` and ``cpu:0`` normalise to the same, type-only, cache key."""
    a = fresh_rotary._normalize_device(torch.device("cpu"))
    b = fresh_rotary._normalize_device(torch.device("cpu", 0))
    assert a == b == torch.device("cpu")
    assert hash(a) == hash(b)


def test_normalize_device_meta_collapses_to_type_only(fresh_rotary):
    """``meta`` normalises to type-only (never gains a spurious index)."""
    a = fresh_rotary._normalize_device(torch.device("meta"))
    assert a == torch.device("meta")


def test_normalize_device_preserves_distinct_indices_cuda(fresh_rotary):
    """``cuda:0`` and ``cuda:1`` must stay distinct after normalisation."""
    a = fresh_rotary._normalize_device(torch.device("cuda:0"))
    b = fresh_rotary._normalize_device(torch.device("cuda:1"))
    assert a != b
    assert a.type == "cuda" and a.index == 0
    assert b.type == "cuda" and b.index == 1


def test_normalize_device_preserves_distinct_indices_npu(fresh_rotary, npu_device_shim):
    """``npu:0`` and ``npu:1`` must stay distinct after normalisation.

    ``torch.device('npu[...]')`` string parsing fails on CI hosts without
    ``torch_npu`` — the ``npu_device_shim`` fixture intercepts just those
    calls with a hermetic stand-in.
    """
    make = npu_device_shim
    a = fresh_rotary._normalize_device(make("npu", 0))
    b = fresh_rotary._normalize_device(make("npu", 1))
    assert a != b
    assert a.type == "npu" and a.index == 0
    assert b.type == "npu" and b.index == 1


def test_normalize_device_rejects_cuda_without_index(fresh_rotary):
    """Bare ``torch.device('cuda')`` must be rejected (was: silent → cuda:0)."""
    with pytest.raises(ValueError, match="explicit index"):
        fresh_rotary._normalize_device(torch.device("cuda"))


def test_normalize_device_rejects_npu_without_index(fresh_rotary, npu_device_shim):
    """Bare ``torch.device('npu')`` must be rejected (was: silent → npu:0)."""
    make = npu_device_shim
    with pytest.raises(ValueError, match="explicit index"):
        fresh_rotary._normalize_device(make("npu"))


def test_engine_style_explicit_index_setter_normalises_cleanly(fresh_rotary, npu_device_shim):
    """The production Engine passes ``torch.device('npu:{rank}')`` — this must
    round-trip through ``_normalize_device`` without raising and preserve
    both type and index.
    """
    make = npu_device_shim
    r0 = fresh_rotary._normalize_device(make("npu", 0))
    assert r0.type == "npu" and r0.index == 0
    r3 = fresh_rotary._normalize_device(make("npu", 3))
    assert r3.type == "npu" and r3.index == 3


# --- cache key & identity -------------------------------------------------

def test_cache_hit_same_params_same_device(fresh_rotary, stub_build):
    """Same (params, device) → cache hit and object identity."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r1 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    info1 = fresh_rotary.get_rope.cache_info()
    r2 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    info2 = fresh_rotary.get_rope.cache_info()

    assert r1 is r2
    assert len(stub_build) == 1
    assert info1.misses == 1 and info1.hits == 0 and info1.currsize == 1
    assert info2.misses == 1 and info2.hits == 1 and info2.currsize == 1


def test_cache_cpu_then_switch_to_gpu_returns_different_object(fresh_rotary, stub_build):
    """CPU → NPU (represented here as ``cuda:0`` for CI portability):
    the second call must return a distinct module on the new device.
    """
    # first: no setter → ambient CPU
    r_cpu = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    # then: switch target device via setter
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r_gpu = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)

    assert r_cpu is not r_gpu
    assert r_cpu.device == torch.device("cpu")
    assert r_gpu.device == torch.device("cuda", 0)
    info = fresh_rotary.get_rope.cache_info()
    assert info.misses == 2 and info.currsize == 2


def test_cache_gpu_then_switch_to_cpu_returns_different_object(fresh_rotary, stub_build):
    """NPU → CPU: the second call must return a distinct CPU module."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r_gpu = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    # clear the explicit setter → resolver falls back to ambient CPU
    fresh_rotary._ROPE_DEVICE = None
    r_cpu = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)

    assert r_gpu is not r_cpu
    assert r_gpu.device == torch.device("cuda", 0)
    assert r_cpu.device == torch.device("cpu")


def test_cache_distinguishes_accelerator_indices(fresh_rotary, stub_build):
    """``cuda:0`` and ``cuda:1`` are distinct cache keys (proxy for npu:0 / npu:1)."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r0 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    fresh_rotary.set_rope_device(torch.device("cuda:1"))
    r1 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)

    assert r0 is not r1
    assert r0.device.index == 0
    assert r1.device.index == 1
    info = fresh_rotary.get_rope.cache_info()
    assert info.misses == 2 and info.currsize == 2


# --- setter / cache_clear semantics --------------------------------------

def test_setter_switch_does_not_mutate_existing(fresh_rotary, stub_build):
    """Rebinding the setter must not migrate or invalidate already-returned modules."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r0 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    original_id = id(r0)
    original_device = r0.device

    # switch away, then back
    fresh_rotary.set_rope_device(torch.device("cuda:1"))
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)

    # r0 itself untouched
    assert id(r0) == original_id
    assert r0.device == original_device

    # returning to cuda:0 hits the same cached object
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r0_again = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    assert r0_again is r0


def test_setter_does_not_call_cache_clear(fresh_rotary, stub_build):
    """``set_rope_device`` is not allowed to flush the cache."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    assert fresh_rotary.get_rope.cache_info().currsize == 1

    fresh_rotary.set_rope_device(torch.device("cuda:1"))
    # cuda:0 entry must survive; only a new call would grow the cache.
    assert fresh_rotary.get_rope.cache_info().currsize == 1


def test_explicit_cache_clear_forces_reconstruction(fresh_rotary, stub_build):
    """After ``get_rope.cache_clear()`` the next call must miss and produce a new object."""
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    r0 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    fresh_rotary.get_rope.cache_clear()
    assert fresh_rotary.get_rope.cache_info().currsize == 0

    r1 = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    assert r0 is not r1
    info = fresh_rotary.get_rope.cache_info()
    assert info.misses == 1 and info.hits == 0 and info.currsize == 1


def test_cache_diagnostics_forwarded_from_get_rope(fresh_rotary):
    """``get_rope.cache_info`` / ``cache_clear`` are aliased from the private cached fn.

    We cannot compare bound-method identity directly (each attribute access
    yields a fresh bound-method wrapper), so we assert behavioural
    equivalence: (1) diagnostics agree on the current state, and
    (2) clearing via ``get_rope`` is observable via the private fn's
    ``cache_info``.
    """
    # after a fresh import the cache is empty from both viewpoints.
    assert fresh_rotary.get_rope.cache_info() == fresh_rotary._get_rope_cached.cache_info()
    # populate the cache
    _ = fresh_rotary.get_rope(128, 128, 32, 10000.0, None)
    info_public = fresh_rotary.get_rope.cache_info()
    info_private = fresh_rotary._get_rope_cached.cache_info()
    assert info_public == info_private
    assert info_public.currsize == 1
    # clearing through the public alias is observed on the private fn
    fresh_rotary.get_rope.cache_clear()
    assert fresh_rotary._get_rope_cached.cache_info().currsize == 0


def test_cache_info_hit_miss_accounting(fresh_rotary, stub_build):
    """Sanity-check the arithmetic on hit / miss counters across a mixed sequence."""
    # miss 1: cpu default
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    # hit 1
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    # miss 2: different params (max_position)
    _ = fresh_rotary.get_rope(128, 128, 64, 1_000_000.0, None)
    # miss 3: different device
    fresh_rotary.set_rope_device(torch.device("cuda:0"))
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)
    # hit 2 on the cuda entry
    _ = fresh_rotary.get_rope(128, 128, 128, 1_000_000.0, None)

    info = fresh_rotary.get_rope.cache_info()
    assert info.misses == 3
    assert info.hits == 2
    assert info.currsize == 3


# --- integration with AttentionLayer ------------------------------------

def test_attention_layer_call_signature_still_works(fresh_rotary):
    """AttentionLayer calls ``get_rope`` with kwargs and ``rope_scaling=None``
    (or a tuple). The public signature must accept both without change.
    """
    # kw-only, rope_scaling=None (Qwen / Llama base case)
    r = fresh_rotary.get_rope(
        head_dim=128, rotary_dim=128, max_position=32, base=10000.0, rope_scaling=None,
    )
    assert r.head_size == 128
    assert r.rotary_dim == 128
    assert r._cos_sin_cache.device == torch.device("cpu")

    # kw-only, rope_scaling=tuple(...) (Llama-3 / YaRN base case)
    scaling = (
        ("rope_type", "llama3"),
        ("factor", 8.0),
        ("low_freq_factor", 1.0),
        ("high_freq_factor", 4.0),
        ("original_max_position_embeddings", 8192),
    )
    r2 = fresh_rotary.get_rope(
        head_dim=128, rotary_dim=128, max_position=32, base=10000.0, rope_scaling=scaling,
    )
    assert isinstance(r2, fresh_rotary.RotaryEmbedding)
    # different scaling → different cache key → different object
    assert r is not r2


def test_public_module_import_still_hides_torch_npu(fresh_rotary):
    """Regression guard for #1 / #2 after the split: no lazy import path
    should have leaked into rotary.py.
    """
    assert "torch_npu" not in sys.modules
    assert "flashinfer" not in sys.modules
