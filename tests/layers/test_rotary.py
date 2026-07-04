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
