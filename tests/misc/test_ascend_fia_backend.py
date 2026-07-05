"""Hermetic tests for the AscendFIABackend across Gates 1.8a → 1.8e.

The class module (``minisgl.attention.ascend_fia``) is torch-free at import
time, so we can exercise real Python semantics — instantiation, method calls,
registry state, metadata construction and the FIA forward call — without
pulling in the CUDA / Ascend runtime.

The engine wiring test uses ``ast``-inspection because
``minisgl.engine.engine`` transitively imports torch + huggingface + kernels.

What's checked here:

Gate 1.8a (scaffolding):
  1. ``AscendFIABackend`` is instantiable (all abstract methods overridden).
  2. All five ``BaseAttnBackend`` interface methods exist on the class.
  3. ``init_capture_graph`` / ``prepare_for_capture`` / ``prepare_for_replay``
     are callable no-ops (return ``None``).
  4. ``SUPPORTED_ATTENTION_BACKENDS`` contains ``"npu_fia"``.
  5. Registering ``"npu_fia"`` does not import ``ascend_fia`` until the factory
     is actually invoked (lazy import).
  6. ``_adjust_config`` maps ``device_type == "npu"`` + ``auto`` → ``"npu_fia"``.
  7. ``_adjust_config`` keeps the existing SM100 / SM90 / other selection on
     CUDA/CPU auto.
  8. Explicit ``fi`` / ``fa`` / ``trtllm`` / ``npu_fia`` are not overridden.
  9. ``ascend_fia`` module has no top-level ``torch_npu`` import.

Gate 1.8c (metadata): single-request BSND ``prepare_metadata`` builder,
block-table stride-then-divide algorithm, field types, ``get_last_indices``,
multi-request rejection.

Gate 1.8e (forward): end-to-end ``forward`` call with a fake ``torch_npu``
module — store_kv-before-FIA ordering, BSND view, prefill / decode mask
construction, KV-dim padding to block boundary, FIA kwargs, tuple return
handling, dynamic-import failure, and the new invariant that ``torch_npu``
only appears inside ``forward``.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
_ASCEND_FIA_PATH = _PYTHON_ROOT / "minisgl" / "attention" / "ascend_fia.py"
_ATTN_INIT_PATH = _PYTHON_ROOT / "minisgl" / "attention" / "__init__.py"
_ENGINE_PATH = _PYTHON_ROOT / "minisgl" / "engine" / "engine.py"


# --------------------------------------------------------------------- helpers


def _ensure_python_root_on_path() -> None:
    p = str(_PYTHON_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_attention() -> object:
    _ensure_python_root_on_path()
    try:
        import minisgl.attention as mod
    except ImportError as exc:  # pragma: no cover — happens only off the repo
        pytest.skip(f"minisgl.attention not importable: {exc}")
    return mod


def _load_ascend_fia_module() -> object:
    _ensure_python_root_on_path()
    try:
        return importlib.import_module("minisgl.attention.ascend_fia")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"minisgl.attention.ascend_fia not importable: {exc}")


def _ascend_fia_source() -> str:
    return _ASCEND_FIA_PATH.read_text()


def _attn_init_source() -> str:
    return _ATTN_INIT_PATH.read_text()


def _engine_source() -> str:
    return _ENGINE_PATH.read_text()


def _engine_tree() -> ast.Module:
    return ast.parse(_engine_source())


def _adjust_config_fn() -> ast.FunctionDef:
    return next(
        node for node in _engine_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_adjust_config"
    )


# --------------------------------- 1 & 2: class + interface completeness ----


def test_ascend_fia_backend_is_instantiable():
    mod = _load_ascend_fia_module()
    cls = mod.AscendFIABackend
    # Abstract classes surface leftover abstract methods here; an empty frozenset
    # proves the skeleton overrode every abstract slot on BaseAttnBackend.
    assert cls.__abstractmethods__ == frozenset(), (
        f"AscendFIABackend still has abstract methods: {cls.__abstractmethods__!r}"
    )
    # Should also actually construct.
    backend = cls(config=None)
    assert backend is not None


def test_ascend_fia_backend_defines_all_five_interfaces():
    mod = _load_ascend_fia_module()
    cls = mod.AscendFIABackend
    for name in (
        "forward",
        "prepare_metadata",
        "init_capture_graph",
        "prepare_for_capture",
        "prepare_for_replay",
    ):
        method = getattr(cls, name, None)
        assert callable(method), f"AscendFIABackend must define {name}()"


# --------------------------------- 3 & 4: no-op semantics -------------------


def test_graph_hooks_return_none():
    mod = _load_ascend_fia_module()
    backend = mod.AscendFIABackend(config=None)
    assert backend.init_capture_graph(max_seq_len=64, bs_list=[1, 2, 4]) is None
    assert backend.prepare_for_capture(batch=None) is None
    assert backend.prepare_for_replay(batch=None) is None


def test_prepare_metadata_returns_none(monkeypatch):
    """Gate 1.8a promised ``prepare_metadata`` was a no-op returning ``None``.
    Gate 1.8c makes it a real builder — the ``-> None`` contract is preserved
    (it mutates ``batch.attn_metadata`` rather than returning a value), but
    the test now feeds a valid single-req fake batch since the empty-batch
    no-op behaviour is gone.
    """
    torch = pytest.importorskip("torch")
    mod = _load_ascend_fia_module()
    backend = mod.AscendFIABackend(config=None)

    # Lightweight fakes — defined below in the Gate 1.8c section.
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    import minisgl.core as core_mod

    class _C:
        page_table = None  # patched below
        page_size = 4

    ctx = _C()
    ctx.page_table = page_table
    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)

    class _R:
        table_idx = 0
        cached_len = 0
        device_len = 4
        @property
        def extend_len(self) -> int:
            return self.device_len - self.cached_len

    class _B:
        def __init__(self):
            self.padded_reqs = [_R()]
            self.attn_metadata = None

    batch = _B()
    assert backend.prepare_metadata(batch) is None
    assert batch.attn_metadata is not None


# --------------------------------- 6: registry membership -------------------


def test_registry_contains_npu_fia():
    mod = _load_attention()
    assert "npu_fia" in mod.SUPPORTED_ATTENTION_BACKENDS.supported_names(), (
        f"expected 'npu_fia' in {mod.SUPPORTED_ATTENTION_BACKENDS.supported_names()!r}"
    )


# --------------------------------- 7: lazy import ---------------------------


def test_registering_npu_fia_does_not_import_ascend_fia_at_attention_import_time():
    """The factory must be lazy: importing ``minisgl.attention`` alone must
    not pull in ``minisgl.attention.ascend_fia``. Only invoking the ``npu_fia``
    factory should trigger the import.
    """
    _ensure_python_root_on_path()
    # Purge both modules so the test genuinely measures lazy semantics.
    sys.modules.pop("minisgl.attention.ascend_fia", None)
    sys.modules.pop("minisgl.attention", None)
    importlib.import_module("minisgl.attention")
    assert "minisgl.attention.ascend_fia" not in sys.modules, (
        "ascend_fia was eagerly imported during minisgl.attention import; "
        "the npu_fia factory must import lazily."
    )
    # And now the factory call must actually cause the import.
    mod = sys.modules["minisgl.attention"]
    factory = mod.SUPPORTED_ATTENTION_BACKENDS["npu_fia"]
    backend = factory(None)  # config=None is fine for the skeleton
    assert backend is not None
    assert "minisgl.attention.ascend_fia" in sys.modules, (
        "invoking the npu_fia factory did not import ascend_fia"
    )


# --------------------------------- 8, 9, 10: _adjust_config wiring ----------
# The engine module transitively imports torch + heavy runtime bits, so use
# ast-level inspection to verify the auto-selection logic.


def test_adjust_config_signature_takes_device_type():
    fn = _adjust_config_fn()
    arg_names = [a.arg for a in fn.args.args]
    assert arg_names == ["config", "device_type"], (
        f"_adjust_config must accept (config, device_type); got {arg_names!r}"
    )


def test_adjust_config_auto_npu_maps_to_npu_fia():
    fn = _adjust_config_fn()
    src = ast.unparse(fn)
    # The npu branch must produce the "npu_fia" backend string.
    assert '"npu_fia"' in src or "'npu_fia'" in src, (
        "_adjust_config must select 'npu_fia' on the npu branch"
    )
    # The npu branch must gate on device_type == "npu".
    # Look for an ast.Compare like device_type == "npu"
    found = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) \
                and isinstance(node.left, ast.Name) and node.left.id == "device_type" \
                and any(isinstance(c, ast.Constant) and c.value == "npu"
                        for c in node.comparators):
            found = True
            break
    assert found, "expected `device_type == 'npu'` inside _adjust_config"


def test_adjust_config_cuda_keeps_original_sm_selection():
    """The non-NPU branch must still consult ``is_sm100_supported`` /
    ``is_sm90_supported`` and produce the original trtllm / fa,fi / fi
    ternary. This guards against a refactor that silently drops CUDA."""
    fn = _adjust_config_fn()
    src = ast.unparse(fn)
    assert "is_sm100_supported" in src, \
        "CUDA branch must still call is_sm100_supported()"
    assert "is_sm90_supported" in src, \
        "CUDA branch must still call is_sm90_supported()"
    assert '"trtllm"' in src or "'trtllm'" in src
    assert '"fa,fi"' in src or "'fa,fi'" in src
    assert '"fi"' in src or "'fi'" in src


def test_adjust_config_only_overrides_when_backend_is_auto():
    """Explicit choices — ``"fi"``, ``"fa"``, ``"trtllm"``, ``"npu_fia"`` — must
    survive intact. The whole selection block must be gated by
    ``config.attention_backend == "auto"``.
    """
    fn = _adjust_config_fn()
    # Find the top-level `if config.attention_backend == "auto":` guard.
    guard = None
    for node in fn.body:
        if isinstance(node, ast.If):
            test_text = ast.unparse(node.test)
            if "config.attention_backend" in test_text and '"auto"' in test_text.replace("'", '"'):
                guard = node
                break
    assert guard is not None, (
        "_adjust_config must gate backend override on "
        "`config.attention_backend == \"auto\"`"
    )
    # Neither branch inside the guard should reference explicit backend names
    # in a way that mutates them.
    body_text = "\n".join(ast.unparse(n) for n in guard.body)
    # The override happens only inside the auto block; the block must contain
    # the override() call. This is a positive assertion that the mutation
    # is scoped to the auto path.
    assert "override" in body_text, \
        "the auto branch must call override(...) to install the resolved backend"


# --------------------------------- 11: no torch_npu at module top -----------


def test_ascend_fia_module_has_no_top_level_torch_npu_import():
    tree = ast.parse(_ascend_fia_source())
    for node in tree.body:  # only top-level nodes
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch_npu"), (
                    f"top-level `import {alias.name}` is forbidden in ascend_fia.py"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch_npu"), (
                f"top-level `from {node.module} import ...` is forbidden in ascend_fia.py"
            )


# --------------------------------- factory + registry cross-check -----------


def test_npu_fia_factory_returns_ascend_fia_backend():
    mod = _load_attention()
    factory = mod.SUPPORTED_ATTENTION_BACKENDS["npu_fia"]
    backend = factory(None)
    from minisgl.attention.ascend_fia import AscendFIABackend
    assert isinstance(backend, AscendFIABackend)


def test_attention_registry_still_has_original_cuda_backends():
    """Sanity check that the new registration didn't overwrite existing CUDA
    entries. The npu_fia addition must be purely additive."""
    mod = _load_attention()
    names = set(mod.SUPPORTED_ATTENTION_BACKENDS.supported_names())
    for expected in ("trtllm", "fi", "fa"):
        assert expected in names, f"registration for {expected!r} was lost"


# =====================================================================
# Gate 1.8c: single-request BSND FIA metadata
#
# All tests below monkeypatch ``minisgl.core.get_global_ctx`` (which the
# lazy import inside ``prepare_metadata`` picks up on the next call). No
# CUDA / NPU is required — the fake ``page_table`` lives on CPU and the
# assertions in ``prepare_metadata`` only require ``block_table.device``
# to equal ``page_table.device``, which holds trivially here.
# =====================================================================


class _FakeCtx:
    def __init__(self, page_table, page_size: int) -> None:
        self.page_table = page_table
        self.page_size = page_size


class _FakeReq:
    """Minimal ``Req`` stand-in — only the fields ``prepare_metadata``
    actually reads. Avoids pulling the real ``Req`` dataclass (which
    validates via ``__post_init__`` on a CPU tensor)."""

    def __init__(self, table_idx: int, cached_len: int, device_len: int) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _FakeBatch:
    def __init__(self, padded_reqs) -> None:
        self.padded_reqs = padded_reqs
        self.attn_metadata = None


def _install_ctx(monkeypatch, page_table, page_size: int) -> _FakeCtx:
    """Patch ``minisgl.core.get_global_ctx`` to yield a ``_FakeCtx``.

    ``ascend_fia.prepare_metadata`` does ``from minisgl.core import
    get_global_ctx`` inside the function body (lazy), so patching the
    attribute on ``minisgl.core`` is picked up on the next call.
    """
    import minisgl.core as core_mod

    ctx = _FakeCtx(page_table=page_table, page_size=page_size)
    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)
    return ctx


def _make_backend():
    mod = _load_ascend_fia_module()
    return mod, mod.AscendFIABackend(config=None)


# --------------------------------- basic single-request paths --------------


def test_prepare_metadata_single_prefill_no_prefix(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # page_size=4, row 0 uses physical pages 0 and 1 (raw slots 0..7).
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=6)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta, mod.FIAMetadata)
    assert meta.query_seq_len == 6
    assert meta.kv_seq_len == 6
    assert meta.actual_seq_lengths == [6]
    assert meta.actual_seq_lengths_kv == [6]
    assert meta.input_layout == "BSND"
    # ceil(6/4) = 2 blocks; stride-4 picks raw slots [0, 4]; /4 → page ids [0, 1]
    assert meta.block_table.tolist() == [[0, 1]]
    assert meta.block_table.shape == (1, 2)
    assert meta.block_table.dtype == torch.int32
    assert meta.block_table.device == page_table.device


def test_prepare_metadata_single_prefill_with_cached_prefix(monkeypatch):
    """`extend_len < device_len` is the partial-prefix-hit prefill case."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # Row uses physical pages 4, 5, 6 → raw slots [16..27]. page_size=4.
    page_table = torch.tensor(
        [[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # cached_len=3 in page 0; device_len=10 → new tokens fill through page 2.
    req = _FakeReq(table_idx=0, cached_len=3, device_len=10)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.query_seq_len == 7  # extend_len
    assert meta.kv_seq_len == 10  # device_len
    assert meta.actual_seq_lengths == [7]
    assert meta.actual_seq_lengths_kv == [10]
    # ceil(10/4)=3; stride-4 picks slots [16, 20, 24]; /4 → [4, 5, 6]
    assert meta.block_table.tolist() == [[4, 5, 6]]
    assert meta.block_table.shape == (1, 3)


def test_prepare_metadata_single_decode(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # Row uses physical pages 2 and 3 → raw slots [8..15].
    page_table = torch.tensor(
        [[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # Decode step: cached_len=5, device_len=6 (1 new token this step).
    req = _FakeReq(table_idx=0, cached_len=5, device_len=6)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.query_seq_len == 1
    assert meta.kv_seq_len == 6
    assert meta.actual_seq_lengths == [1]
    assert meta.actual_seq_lengths_kv == [6]
    # ceil(6/4)=2; stride-4 picks slots [8, 12]; /4 → [2, 3]
    assert meta.block_table.tolist() == [[2, 3]]


# --------------------------------- block-table shape / algorithm ----------


def test_block_table_multi_page(monkeypatch):
    """A three-page KV must yield block_table shape [1, 3]."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=12)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.block_table.shape == (1, 3)
    assert meta.block_table.tolist() == [[0, 1, 2]]


def test_block_table_non_contiguous_page_ids(monkeypatch):
    """Physical pages allocated out of logical order (e.g. [2, 0]) must
    survive the raw-slot → page-id conversion intact."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # Row uses physical page 2 first (raw slots 8..11), then page 0
    # (raw slots 0..3). This is a real cache-allocator pattern under
    # fragmentation.
    page_table = torch.tensor(
        [[8, 9, 10, 11, 0, 1, 2, 3]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=8)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    # stride-4 picks first slot of each page: [8, 0]; /4 → [2, 0]
    assert meta.block_table.tolist() == [[2, 0]]


def test_block_table_stride_then_divide_order(monkeypatch):
    """Guard the ordering: (stride the row, then divide) — divide-first
    on the full row would produce wrong page ids for non-first-of-page
    slots. The test sets non-first slots to a poison value that would
    surface if the algorithm ever regressed."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # First slot of each page correctly encodes the page id * page_size.
    # Non-first slots are poison = 99 — if the implementation divided the
    # whole row by page_size and then picked stride, it would look at
    # 99 // 4 == 24 and produce [24, ...] instead of [0, 1].
    page_table = torch.tensor(
        [[0, 99, 99, 99, 4, 99, 99, 99]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=8)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.block_table.tolist() == [[0, 1]]


# --------------------------------- field types ---------------------------


def test_actual_seq_lengths_are_python_lists(monkeypatch):
    """FIA's actual_seq_lengths* are Python lists, not tensors — this
    lets torch_npu skip a device round-trip. Regressing this to a tensor
    is a silent perf pitfall we guard against."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=4)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta.actual_seq_lengths, list)
    assert isinstance(meta.actual_seq_lengths_kv, list)
    assert all(isinstance(x, int) for x in meta.actual_seq_lengths)
    assert all(isinstance(x, int) for x in meta.actual_seq_lengths_kv)


def test_input_layout_is_bsnd(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=4)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    assert batch.attn_metadata.input_layout == "BSND"


# --------------------------------- get_last_indices ----------------------


def test_get_last_indices_prefill(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # extend_len == 6 → last index in the flat Q buffer == 5
    req = _FakeReq(table_idx=0, cached_len=0, device_len=6)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    idx = batch.attn_metadata.get_last_indices(1)
    assert idx.tolist() == [5]
    assert idx.dtype == torch.int32
    assert idx.shape == (1,)
    assert idx.device == page_table.device


def test_get_last_indices_decode(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    _install_ctx(monkeypatch, page_table, page_size=4)

    # decode step: extend_len == 1 → last index == 0
    req = _FakeReq(table_idx=0, cached_len=3, device_len=4)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    idx = batch.attn_metadata.get_last_indices(1)
    assert idx.tolist() == [0]
    assert idx.dtype == torch.int32


def test_get_last_indices_bs_two_equal_length(monkeypatch):
    """Gate 2.2c: equal-length B=2 → last indices are the flat-buffer offsets
    of the last query token per request.
    """
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=4),
        _FakeReq(table_idx=1, cached_len=0, device_len=4),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    # Flat layout is [r0_t0..r0_t3, r1_t0..r1_t3]; last-per-req = [3, 7].
    idx = batch.attn_metadata.get_last_indices(2)
    assert idx.tolist() == [3, 7]
    assert idx.dtype == torch.int32
    assert idx.device == page_table.device


# --------------------------------- Gate 2.2c: equal-length B>=1 ----------


def test_prepare_metadata_equal_length_b2_prefill(monkeypatch):
    """Gate 2.2c: equal-length B=2 prefill must build a shared metadata."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # Two independent physical-page rows: req0 → pages [0,1], req1 → pages [4,5].
    page_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [16, 17, 18, 19, 20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=6),
        _FakeReq(table_idx=1, cached_len=0, device_len=6),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta, mod.FIAMetadata)
    assert meta.batch_size == 2
    assert meta.query_seq_len == 6
    assert meta.kv_seq_len == 6
    assert meta.actual_seq_lengths == [6, 6]
    assert meta.actual_seq_lengths_kv == [6, 6]
    assert meta.input_layout == "BSND"
    # ceil(6/4) = 2 blocks; stride-4 picks slots; /4 → page ids per row.
    assert meta.block_table.tolist() == [[0, 1], [4, 5]]
    assert meta.block_table.shape == (2, 2)
    assert meta.block_table.dtype == torch.int32
    assert meta.block_table.device == page_table.device


def test_prepare_metadata_equal_length_b2_decode(monkeypatch):
    """Gate 2.2c: equal-length B=2 decode (query_seq_len=1)."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [
            [8, 9, 10, 11, 12, 13, 14, 15],
            [40, 41, 42, 43, 44, 45, 46, 47],
        ],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # Both requests: 1 new token this step, KV of length 6.
    reqs = [
        _FakeReq(table_idx=0, cached_len=5, device_len=6),
        _FakeReq(table_idx=1, cached_len=5, device_len=6),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.batch_size == 2
    assert meta.query_seq_len == 1
    assert meta.kv_seq_len == 6
    assert meta.actual_seq_lengths == [1, 1]
    assert meta.actual_seq_lengths_kv == [6, 6]
    # Row 0 uses raw slots [8,12] → pages [2,3]; row 1 uses [40,44] → [10,11].
    assert meta.block_table.tolist() == [[2, 3], [10, 11]]


def test_prepare_metadata_block_table_rows_independent(monkeypatch):
    """Gate 2.2c: each row of block_table must be sourced from its own
    page_table row — no cross-request contamination."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    # Row 0 uses fragmented physical pages [2, 0]; row 1 uses [1, 3].
    page_table = torch.tensor(
        [
            [8, 9, 10, 11, 0, 1, 2, 3],
            [4, 5, 6, 7, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=8),
        _FakeReq(table_idx=1, cached_len=0, device_len=8),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    # Row 0: stride-4 picks [8, 0] → [2, 0]. Row 1: picks [4, 12] → [1, 3].
    assert batch.attn_metadata.block_table.tolist() == [[2, 0], [1, 3]]


def test_prepare_metadata_rejects_ragged_extend_len(monkeypatch):
    """Gate 2.2c required rejection of ragged ``extend_len``; Gate 2.2f
    relaxes this to accept the cached_len==0 case. This test now verifies
    that acceptance: a batch with mixed extend_len but zero cached_len on
    every request is a valid ragged-prefill batch.

    The strict rejection is retained by
    :func:`test_gate22f_prepare_metadata_rejects_ragged_with_cached` below,
    which covers the mixed-prefix-hit ragged case.
    """
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=4),   # extend_len=4
        _FakeReq(table_idx=1, cached_len=0, device_len=3),   # extend_len=3
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    # Gate 2.2f: accepted.
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata
    assert meta.query_seq_lens == [4, 3]
    assert meta.max_query_len == 4
    assert meta.query_seq_len is None  # ragged sentinel


def test_prepare_metadata_rejects_ragged_cached_len(monkeypatch):
    """Gate 2.2c: mismatched ``cached_len`` must raise NotImplementedError,
    even when ``extend_len`` matches. This is the mixed-prefix-hit case."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # Both extend_len=2 but cached_len differs (0 vs 4).
    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=2),
        _FakeReq(table_idx=1, cached_len=4, device_len=6),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    with pytest.raises(NotImplementedError) as excinfo:
        backend.prepare_metadata(batch)
    assert "ragged" in str(excinfo.value)


# --------------------------------- Gate 1.8a invariants (still hold) -----


def test_gate18c_module_torch_npu_only_inside_forward():
    """Gate 1.8e permits ``torch_npu`` inside ``forward`` (dynamic import +
    single FIA call), but nowhere else. Walk the AST and require that every
    ``torch_npu`` reference — import, name lookup, or attribute chain — sits
    inside the ``forward`` function body of ``AscendFIABackend``.
    """
    tree = ast.parse(_ascend_fia_source())

    forward_fn = None
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "AscendFIABackend":
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    forward_fn = item
                    break
    assert forward_fn is not None, "AscendFIABackend.forward not found"

    forward_nodes = set(id(n) for n in ast.walk(forward_fn))

    for node in ast.walk(tree):
        offending = False
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("torch_npu"):
                    offending = True
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("torch_npu"):
                offending = True
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "torch_npu":
                offending = True
        elif isinstance(node, ast.Name):
            if node.id == "torch_npu":
                offending = True

        if offending:
            assert id(node) in forward_nodes, (
                "torch_npu references are only allowed inside "
                "AscendFIABackend.forward"
            )


# =====================================================================
# Gate 1.8e: single-request BSND FIA forward
#
# All tests below stub ``minisgl.core.get_global_ctx`` with a fake context
# that exposes a fake ``kv_cache`` (recording ``store_kv`` + returning fake
# BnNBsD tensors from ``k_cache`` / ``v_cache``), and inject a fake
# ``torch_npu`` module into ``sys.modules`` so ``forward`` picks it up on the
# next dynamic ``import torch_npu``.
#
# No real Ascend runtime or CUDA runtime is exercised — every tensor lives on
# CPU. The fake FIA op records its inputs and returns a well-shaped
# ``(attention_out, softmax_lse)`` tuple.
# =====================================================================


import types  # noqa: E402 — placed here so the Gate 1.8e block reads as a unit


class _RecordingKVCache:
    """Fake KV cache exercising the exact surface ``forward`` touches.

    * ``store_kv(k, v, out_loc, layer_id)`` — appends the call to
      ``store_kv_calls`` so tests can assert ordering / arguments.
    * ``k_cache(layer_id)`` / ``v_cache(layer_id)`` — return per-layer tensors
      that stand in for the paged BnNBsD cache. ``forward`` must pass these
      through verbatim; we use ``is`` identity in the tests to prove no
      permute / contiguous / clone happened.
    """

    def __init__(self, num_pages: int, num_kv_heads: int, page_size: int,
                 head_dim: int, num_layers: int, dtype) -> None:
        import torch as _t

        self.store_kv_calls = []
        # Distinct per-layer tensors so a test can assert that ``layer_id`` is
        # threaded through correctly.
        self._k_caches = [
            _t.randn((num_pages, num_kv_heads, page_size, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self._v_caches = [
            _t.randn((num_pages, num_kv_heads, page_size, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]

    def store_kv(self, k, v, out_loc, layer_id):
        self.store_kv_calls.append(
            {"k": k, "v": v, "out_loc": out_loc, "layer_id": layer_id}
        )

    def k_cache(self, layer_id):
        return self._k_caches[layer_id]

    def v_cache(self, layer_id):
        return self._v_caches[layer_id]


class _FakeCtxFIA:
    def __init__(self, page_table, page_size, kv_cache) -> None:
        self.page_table = page_table
        self.page_size = page_size
        self.kv_cache = kv_cache


class _FakeBatchFIA:
    def __init__(self, padded_reqs, out_loc) -> None:
        self.padded_reqs = padded_reqs
        self.out_loc = out_loc
        self.attn_metadata = None


def _install_fake_torch_npu(monkeypatch, calls, out_shape, out_dtype, out_device):
    """Install a fake ``torch_npu`` in ``sys.modules`` with a recording
    ``npu_fused_infer_attention_score`` that returns ``(attention_out,
    softmax_lse)`` — matching FIA's real inference-mode arity."""
    import torch as _t

    fake = types.ModuleType("torch_npu")

    def _fake_fia(query, key, value, **kwargs):
        calls.append({
            "query": query,
            "key": key,
            "value": value,
            "kwargs": kwargs,
        })
        attention_out = _t.zeros(out_shape, dtype=out_dtype, device=out_device)
        softmax_lse = _t.empty((0,), dtype=_t.float32, device=out_device)
        return (attention_out, softmax_lse)

    fake.npu_fused_infer_attention_score = _fake_fia
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return fake


def _prime_forward(monkeypatch, *, query_seq_len, kv_seq_len,
                    num_heads=4, num_kv_heads=2, head_dim=8, page_size=4,
                    num_pages=8, num_layers=2, layer_id=0):
    """Build a wired-up (backend, batch, ctx, fake_npu_calls) tuple.

    ``prepare_metadata`` is invoked so ``batch.attn_metadata`` is a real
    :class:`FIAMetadata`; then a fake ``torch_npu`` is injected so
    ``forward`` can be called without any real Ascend runtime.
    """
    import torch as _t

    mod, backend = _make_backend()

    # ``device_len`` is total KV, ``cached_len`` = kv_seq_len - query_seq_len.
    cached_len = kv_seq_len - query_seq_len
    device_len = kv_seq_len

    # page_table row 0 uses contiguous physical pages 0..N-1 → simplifies
    # the block_table assertions to consecutive small ints.
    num_blocks = (kv_seq_len + page_size - 1) // page_size
    row = list(range(num_blocks * page_size))
    # Pad to at least device_len (metadata slices ``: num_blocks * page_size``).
    page_table = _t.tensor([row], dtype=_t.int32)

    kv_cache = _RecordingKVCache(
        num_pages=num_pages, num_kv_heads=num_kv_heads,
        page_size=page_size, head_dim=head_dim,
        num_layers=num_layers, dtype=_t.float32,
    )

    ctx = _FakeCtxFIA(page_table=page_table, page_size=page_size, kv_cache=kv_cache)

    import minisgl.core as core_mod

    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)

    req = _FakeReq(table_idx=0, cached_len=cached_len, device_len=device_len)
    # ``out_loc`` is opaque to forward — just a marker object we can identity-check.
    out_loc = _t.arange(query_seq_len, dtype=_t.int32)
    batch = _FakeBatchFIA(padded_reqs=[req], out_loc=out_loc)

    backend.prepare_metadata(batch)

    q = _t.randn((query_seq_len, num_heads, head_dim), dtype=_t.float32)
    k = _t.randn((query_seq_len, num_kv_heads, head_dim), dtype=_t.float32)
    v = _t.randn((query_seq_len, num_kv_heads, head_dim), dtype=_t.float32)

    calls = []
    _install_fake_torch_npu(
        monkeypatch, calls,
        out_shape=(1, query_seq_len, num_heads, head_dim),
        out_dtype=_t.float32, out_device=q.device,
    )

    return {
        "mod": mod,
        "backend": backend,
        "batch": batch,
        "ctx": ctx,
        "kv_cache": kv_cache,
        "q": q, "k": k, "v": v,
        "layer_id": layer_id,
        "calls": calls,
        "num_blocks": num_blocks,
        "page_size": page_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "query_seq_len": query_seq_len,
        "kv_seq_len": kv_seq_len,
        "cached_len": cached_len,
    }


# --------------------------------- 1. store_kv before FIA ------------------


def test_forward_calls_store_kv_before_fia(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=4, kv_seq_len=4, layer_id=1)
    backend, batch = setup["backend"], setup["batch"]

    # Cross-reference the two call recorders by wall-clock order — but since
    # this test runs synchronously, a simpler check is: at the moment the fake
    # FIA op fires, ``store_kv_calls`` already has exactly one entry.
    kv_cache = setup["kv_cache"]
    fia_calls = setup["calls"]

    original_fake = sys.modules["torch_npu"].npu_fused_infer_attention_score

    def _wrapped(*args, **kwargs):
        # When FIA runs, store_kv must already have been called exactly once.
        assert len(kv_cache.store_kv_calls) == 1, (
            "store_kv must run before FIA"
        )
        return original_fake(*args, **kwargs)

    sys.modules["torch_npu"].npu_fused_infer_attention_score = _wrapped

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    assert len(fia_calls) == 1
    assert len(kv_cache.store_kv_calls) == 1
    call = kv_cache.store_kv_calls[0]
    assert call["k"] is setup["k"]
    assert call["v"] is setup["v"]
    assert call["out_loc"] is batch.out_loc
    assert call["layer_id"] == setup["layer_id"]


# --------------------------------- 2. q view to BSND -----------------------


def test_forward_views_q_to_bsnd(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=3, kv_seq_len=3)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    query = setup["calls"][0]["query"]
    # [1, T, H, D]
    assert tuple(query.shape) == (
        1, setup["query_seq_len"], setup["num_heads"], setup["head_dim"],
    )
    # unsqueeze is a non-copy view → underlying storage is shared with the
    # flat q. ``data_ptr`` equality is the crispest form of this check.
    assert query.data_ptr() == setup["q"].data_ptr()


# --------------------------------- 3. Prefill no-prefix mask ---------------


def test_forward_prefill_no_prefix_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=4, kv_seq_len=4, page_size=4)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    assert mask is not None
    # num_blocks == 1, page_size == 4 → padded KV dim == 4 (already equals kv_seq_len).
    assert tuple(mask.shape) == (4, 4)
    # No cached prefix ⇒ standard upper-triangular causal mask (diagonal=1).
    expected = torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1)
    assert torch.equal(mask, expected)


# --------------------------------- 4. Prefill cached-prefix mask -----------


def test_forward_prefill_cached_prefix_offset_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    # cached_len=14, query_seq_len=4, kv_seq_len=18 → the Gate 1.8d shape.
    setup = _prime_forward(monkeypatch, query_seq_len=4, kv_seq_len=18, page_size=16)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    assert mask is not None
    # num_blocks == ceil(18/16) == 2 → padded KV dim == 32.
    assert tuple(mask.shape) == (4, 32)

    # Visible KV per row (up to the true kv_seq_len == 18) must match
    # [15, 16, 17, 18] — the Gate 1.8d visibility spec.
    visible = (~mask[:, :18]).sum(dim=1).tolist()
    assert visible == [15, 16, 17, 18]


# --------------------------------- 5. Mask KV padded to block boundary ----


def test_forward_mask_kv_dim_padded_to_block_boundary(monkeypatch):
    torch = pytest.importorskip("torch")
    # Deliberately choose a kv_seq_len that is NOT a multiple of page_size so
    # the padding is non-trivial.
    setup = _prime_forward(monkeypatch, query_seq_len=3, kv_seq_len=5, page_size=4)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    # num_blocks == ceil(5/4) == 2; padded == 8. This is the FIA-tiling
    # requirement surfaced by the Gate 1.8d probe.
    assert tuple(mask.shape) == (3, 8)


# --------------------------------- 6. Padding columns all masked ----------


def test_forward_padding_columns_are_all_true(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=3, kv_seq_len=5, page_size=4)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    # Columns >= kv_seq_len == 5 are the "padding" columns — they must be
    # entirely True (== masked out) so they can never contribute to any row.
    pad_cols = mask[:, 5:]
    assert bool(pad_cols.all().item()), (
        "padding columns must all be True (== masked out)"
    )


# --------------------------------- 7. Decode atten_mask is None -----------


def test_forward_decode_mask_is_none(monkeypatch):
    torch = pytest.importorskip("torch")
    # query_seq_len == 1 is the decode path.
    setup = _prime_forward(monkeypatch, query_seq_len=1, kv_seq_len=6, page_size=4)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    assert setup["calls"][0]["kwargs"]["atten_mask"] is None


# --------------------------------- 8. FIA kwargs correct ------------------


def test_forward_fia_kwargs_are_correct(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(
        monkeypatch, query_seq_len=6, kv_seq_len=6, page_size=4,
        num_heads=4, num_kv_heads=2, head_dim=8,
    )
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    kwargs = setup["calls"][0]["kwargs"]

    # block_table is what metadata built (page 0, page 1 for a 6-token seq at
    # page_size=4 → ceil(6/4)=2 pages).
    assert kwargs["block_table"] is batch.attn_metadata.block_table
    # Python-list scalar lengths (Gate 1.8c invariant preserved through 1.8e).
    assert kwargs["actual_seq_lengths"] == [6]
    assert kwargs["actual_seq_lengths_kv"] == [6]
    # Head counts derived from tensor shapes, not stashed constants.
    assert kwargs["num_heads"] == setup["num_heads"]
    assert kwargs["num_key_value_heads"] == setup["num_kv_heads"]
    # scale must be 1/sqrt(head_dim) — spelled ``scale`` (NOT ``scale_value``)
    # to match the aclnn v3 binding.
    assert "scale" in kwargs and "scale_value" not in kwargs
    import math
    assert kwargs["scale"] == pytest.approx(1.0 / math.sqrt(setup["head_dim"]))
    # BSND single-request path.
    assert kwargs["input_layout"] == "BSND"
    assert kwargs["block_size"] == setup["page_size"]
    assert kwargs["sparse_mode"] == 0


# --------------------------------- 9. K/V passed verbatim ------------------


def test_forward_kv_cache_tensors_passed_by_identity(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=4, kv_seq_len=4, layer_id=1)
    backend, batch = setup["backend"], setup["batch"]

    backend.forward(setup["q"], setup["k"], setup["v"],
                    layer_id=setup["layer_id"], batch=batch)

    call = setup["calls"][0]
    # The paged BnNBsD caches must be forwarded without permute / contiguous /
    # clone. ``is`` identity is the tightest check.
    assert call["key"] is setup["kv_cache"].k_cache(setup["layer_id"])
    assert call["value"] is setup["kv_cache"].v_cache(setup["layer_id"])


# --------------------------------- 10. FIA tuple second item ignored ------


def test_forward_ignores_second_tuple_item(monkeypatch):
    """FIA returns ``(attention_out, softmax_lse)`` in inference mode;
    softmax_lse is empty and must not be consumed by the backend."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=2, kv_seq_len=2)
    backend, batch = setup["backend"], setup["batch"]

    import torch as _t

    poisoned_lse = _t.full((3,), float("nan"), dtype=_t.float32)
    expected_out = _t.zeros(
        (1, setup["query_seq_len"], setup["num_heads"], setup["head_dim"]),
        dtype=_t.float32,
    )
    expected_out.fill_(7.0)

    def _fake_fia_two_items(query, key, value, **kwargs):
        setup["calls"].append({
            "query": query, "key": key, "value": value, "kwargs": kwargs,
        })
        return (expected_out, poisoned_lse)

    sys.modules["torch_npu"].npu_fused_infer_attention_score = _fake_fia_two_items

    out = backend.forward(setup["q"], setup["k"], setup["v"],
                          layer_id=setup["layer_id"], batch=batch)

    # Result must be the first tuple item, reshaped, not the LSE.
    assert not bool(torch.isnan(out).any().item())
    assert torch.allclose(out, expected_out.view(setup["q"].shape))


# --------------------------------- 11. Return shape == q shape ------------


def test_forward_return_shape_matches_q(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=5, kv_seq_len=9, page_size=4)
    backend, batch = setup["backend"], setup["batch"]

    out = backend.forward(setup["q"], setup["k"], setup["v"],
                          layer_id=setup["layer_id"], batch=batch)

    assert tuple(out.shape) == tuple(setup["q"].shape)


# --------------------------------- 12. Dynamic import failure -------------


def test_forward_raises_runtimeerror_when_torch_npu_missing(monkeypatch):
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=2, kv_seq_len=2)
    backend, batch = setup["backend"], setup["batch"]

    # ``sys.modules[name] = None`` is Python's canonical way to force
    # ``import name`` to raise ``ImportError`` on the next attempt.
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    with pytest.raises(RuntimeError) as excinfo:
        backend.forward(setup["q"], setup["k"], setup["v"],
                        layer_id=setup["layer_id"], batch=batch)
    msg = str(excinfo.value)
    assert "torch_npu" in msg
    assert "npu_fia" in msg


# --------------------------------- 13. Multi-request still rejected -------


def test_forward_flat_query_shape_mismatch_rejected(monkeypatch):
    """Gate 2.2c / 2.2f: if a caller mutates ``padded_reqs`` between
    ``prepare_metadata`` and ``forward`` so that the flat query no longer
    matches ``sum(query_seq_lens)`` (== ``query_offsets[-1]``), ``forward``
    must refuse."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=2, kv_seq_len=2)
    backend, batch = setup["backend"], setup["batch"]

    # Poison the metadata so it claims B=2 (4 tokens total) while the flat q
    # is still B=1 (2 tokens).
    batch.attn_metadata.actual_seq_lengths = [2, 2]
    batch.attn_metadata.actual_seq_lengths_kv = [2, 2]
    batch.attn_metadata.batch_size = 2
    batch.attn_metadata.query_seq_lens = [2, 2]
    batch.attn_metadata.kv_seq_lens = [2, 2]
    batch.attn_metadata.max_query_len = 2
    batch.attn_metadata.query_offsets = [0, 2, 4]

    with pytest.raises(ValueError) as excinfo:
        backend.forward(setup["q"], setup["k"], setup["v"],
                        layer_id=setup["layer_id"], batch=batch)
    msg = str(excinfo.value)
    assert "flat query" in msg
    assert "sum(query_seq_lens)" in msg


# --------------------------------- 14. Module top-level no torch_npu ------


def test_gate18e_module_no_top_level_torch_npu():
    """Gate 1.8e still forbids top-level ``import torch_npu`` — the dynamic
    import must sit inside ``forward``."""
    tree = ast.parse(_ascend_fia_source())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch_npu"), (
                    f"top-level `import {alias.name}` is forbidden"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch_npu"), (
                f"top-level `from {node.module} import ...` is forbidden"
            )


# =====================================================================
# Gate 2.2c: equal-length B>=1 forward
#
# The fake torch_npu records the query BSND shape and the FIA kwargs — we
# assert the metadata is threaded through correctly and that the shared
# causal mask is broadcast across the batch (single 2-D mask, not per-req).
# =====================================================================


def _prime_forward_b2(monkeypatch, *, query_seq_len, kv_seq_len,
                       num_heads=4, num_kv_heads=2, head_dim=8, page_size=4,
                       num_pages=16, num_layers=2, layer_id=0):
    """Two-request equal-length variant of :func:`_prime_forward`.

    Row 0 uses physical pages [0..num_blocks-1]; row 1 uses
    [num_blocks..2*num_blocks-1]. Both rows are equal-length so
    ``prepare_metadata`` accepts them.
    """
    import torch as _t

    mod, backend = _make_backend()

    cached_len = kv_seq_len - query_seq_len
    device_len = kv_seq_len

    num_blocks = (kv_seq_len + page_size - 1) // page_size
    row0 = list(range(num_blocks * page_size))
    # Row 1 uses a disjoint physical page range so tests can distinguish rows.
    offset = num_blocks * page_size
    row1 = list(range(offset, offset + num_blocks * page_size))
    page_table = _t.tensor([row0, row1], dtype=_t.int32)

    kv_cache = _RecordingKVCache(
        num_pages=num_pages, num_kv_heads=num_kv_heads,
        page_size=page_size, head_dim=head_dim,
        num_layers=num_layers, dtype=_t.float32,
    )
    ctx = _FakeCtxFIA(page_table=page_table, page_size=page_size, kv_cache=kv_cache)

    import minisgl.core as core_mod
    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)

    reqs = [
        _FakeReq(table_idx=0, cached_len=cached_len, device_len=device_len),
        _FakeReq(table_idx=1, cached_len=cached_len, device_len=device_len),
    ]
    # ``out_loc`` layout: req0 raw slots then req1 raw slots, flat.
    out_loc = _t.tensor(
        row0[cached_len:cached_len + query_seq_len]
        + row1[cached_len:cached_len + query_seq_len],
        dtype=_t.int32,
    )
    batch = _FakeBatchFIA(padded_reqs=reqs, out_loc=out_loc)

    backend.prepare_metadata(batch)

    total_tokens = 2 * query_seq_len
    q = _t.randn((total_tokens, num_heads, head_dim), dtype=_t.float32)
    k = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)
    v = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)

    calls = []
    _install_fake_torch_npu(
        monkeypatch, calls,
        out_shape=(2, query_seq_len, num_heads, head_dim),
        out_dtype=_t.float32, out_device=q.device,
    )

    return {
        "backend": backend, "batch": batch, "ctx": ctx, "kv_cache": kv_cache,
        "q": q, "k": k, "v": v, "layer_id": layer_id, "calls": calls,
        "num_blocks": num_blocks, "page_size": page_size,
        "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "query_seq_len": query_seq_len, "kv_seq_len": kv_seq_len,
        "cached_len": cached_len, "row0": row0, "row1": row1,
    }


def test_forward_b2_prefill_query_bsnd_shape(monkeypatch):
    """Gate 2.2c: flat q [B*S, Hq, D] → BSND [B=2, S, Hq, D]."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=4, kv_seq_len=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    query = setup["calls"][0]["query"]
    assert tuple(query.shape) == (
        2, setup["query_seq_len"], setup["num_heads"], setup["head_dim"],
    )


def test_forward_b2_prefill_kwargs(monkeypatch):
    """Gate 2.2c: FIA kwargs mirror the equal-length metadata."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=6, kv_seq_len=6)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    kwargs = setup["calls"][0]["kwargs"]
    assert kwargs["actual_seq_lengths"] == [6, 6]
    assert kwargs["actual_seq_lengths_kv"] == [6, 6]
    # Row 0 → pages [0, 1]; row 1 → pages [2, 3] (offset 8//4=2).
    assert kwargs["block_table"].tolist() == [[0, 1], [2, 3]]
    assert tuple(kwargs["block_table"].shape) == (2, 2)
    assert kwargs["input_layout"] == "BSND"
    assert kwargs["num_heads"] == setup["num_heads"]
    assert kwargs["num_key_value_heads"] == setup["num_kv_heads"]


def test_forward_b2_prefill_shared_causal_mask(monkeypatch):
    """Gate 2.2c: prefill uses a shared [S, padded_kv_len] causal mask; the
    shared cached prefix is visible to every row."""
    torch = pytest.importorskip("torch")
    # cached_len=4, extend_len=4, kv_seq_len=8, page_size=4 → num_blocks=2,
    # padded_kv_len=8. Mask must reveal 5,6,7,8 columns per row.
    setup = _prime_forward_b2(monkeypatch, query_seq_len=4, kv_seq_len=8, page_size=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    assert mask is not None
    # Shape is [S, padded_kv_len], NOT [B, S, padded_kv_len] — shared broadcast.
    assert tuple(mask.shape) == (4, 8)
    visible = (~mask).sum(dim=1).tolist()
    assert visible == [5, 6, 7, 8]


def test_forward_b2_decode_atten_mask_none(monkeypatch):
    """Gate 2.2c: decode (S==1) sets atten_mask=None regardless of B."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=1, kv_seq_len=6, page_size=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    kwargs = setup["calls"][0]["kwargs"]
    assert kwargs["atten_mask"] is None
    query = setup["calls"][0]["query"]
    assert tuple(query.shape) == (2, 1, setup["num_heads"], setup["head_dim"])
    assert kwargs["actual_seq_lengths"] == [1, 1]
    assert kwargs["actual_seq_lengths_kv"] == [6, 6]


def test_forward_b2_store_kv_receives_full_out_loc(monkeypatch):
    """Gate 2.2c: store_kv is called once per layer with the concatenated
    flat ``batch.out_loc``; per-request slot ranges are preserved so pages
    stay isolated. We verify the ranges point into disjoint physical pages
    (row0 vs row1)."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=4, kv_seq_len=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    assert len(setup["kv_cache"].store_kv_calls) == 1
    call = setup["kv_cache"].store_kv_calls[0]
    assert call["k"] is setup["k"]
    assert call["v"] is setup["v"]
    assert call["out_loc"] is setup["batch"].out_loc
    # First half comes from row0's page range, second half from row1's.
    row0_ids = set(setup["row0"])
    row1_ids = set(setup["row1"])
    out_loc_list = call["out_loc"].tolist()
    first_half = out_loc_list[: setup["query_seq_len"]]
    second_half = out_loc_list[setup["query_seq_len"]:]
    assert all(s in row0_ids for s in first_half)
    assert all(s in row1_ids for s in second_half)
    assert row0_ids.isdisjoint(row1_ids)


def test_forward_b2_return_shape_matches_flat_q(monkeypatch):
    """Gate 2.2c: forward must reshape FIA's [B, S, Hq, D] back to the
    caller's flat [B*S, Hq, D]."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=3, kv_seq_len=3)
    out = setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                                    layer_id=setup["layer_id"], batch=setup["batch"])
    assert tuple(out.shape) == tuple(setup["q"].shape)


# --------------------------------- Gate 2.2c: CUDA registration guard ----


def test_gate22c_cuda_backends_still_registered():
    """Gate 2.2c must not disturb CUDA backend registration — the lifting of
    the B=1 hard limit is Ascend-only. Any accidental import churn that
    dropped ``trtllm`` / ``fi`` / ``fa`` from the factory registry would
    surface here.
    """
    mod = _load_attention()
    names = set(mod.SUPPORTED_ATTENTION_BACKENDS.supported_names())
    for expected in ("trtllm", "fi", "fa", "npu_fia"):
        assert expected in names, f"registration for {expected!r} was lost"


def test_gate22c_cuda_backend_metadata_types_unchanged():
    """The CUDA backends still bundle their own metadata dataclasses that
    keep ``get_last_indices`` on ``cu_seqlens_q`` — untouched by 2.2c.

    Read the sources directly (torch-free) so the test is hermetic on hosts
    that don't have torch installed.
    """
    for name in ("fa", "fi", "trtllm"):
        src = (_PYTHON_ROOT / "minisgl" / "attention" / f"{name}.py").read_text()
        assert "get_last_indices" in src, (
            f"{name}: get_last_indices was unexpectedly removed"
        )
        assert "cu_seqlens_q" in src, (
            f"{name}: cu_seqlens_q was unexpectedly removed"
        )


def test_gate22c_only_ascend_fia_touched_batch_limit():
    """Guard: the ``batch size 1 only`` phrase must be absent from
    ascend_fia.py after 2.2c. It also must never leak into another backend.
    """
    src = _ascend_fia_source()
    assert "batch size 1 only" not in src, (
        "Gate 2.2c must remove the legacy 'batch size 1 only' hard limit"
    )


# =====================================================================
# Gate 2.2f: ragged prefill (all cached_len==0, varied extend_len)
#
# Metadata expansion + prepare_metadata acceptance + pack/unpack in forward.
# Fake torch_npu records the packed BSND query, per-batch mask and kwargs.
# =====================================================================


def test_gate22f_prepare_metadata_ragged_prefill_lengths_4_2(monkeypatch):
    """Gate 2.2f: ragged prefill B=2 with extend_len=[4,2], both cached_len==0."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],     # req0: page 0 first raw slot 0 → id 0
            [16, 17, 18, 19, 20, 21, 22, 23],  # req1: page 4 first raw slot 16 → id 4
        ],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=4),  # extend=4 → 1 block
        _FakeReq(table_idx=1, cached_len=0, device_len=2),  # extend=2 → 1 block
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta, mod.FIAMetadata)
    assert meta.batch_size == 2
    assert meta.query_seq_lens == [4, 2]
    assert meta.kv_seq_lens == [4, 2]
    assert meta.max_query_len == 4
    assert meta.query_offsets == [0, 4, 6]
    assert meta.actual_seq_lengths == [4, 2]
    assert meta.actual_seq_lengths_kv == [4, 2]
    # Equal-length shortcuts must be None under ragged.
    assert meta.query_seq_len is None
    assert meta.kv_seq_len is None
    # block_table: each row has ceil(kv/page_size)=1 block; padded to
    # max_blocks=1 so [[0], [4]].
    assert meta.block_table.tolist() == [[0], [4]]
    assert meta.block_table.shape == (2, 1)


def test_gate22f_prepare_metadata_pads_block_table_rows(monkeypatch):
    """Gate 2.2f: request with fewer blocks than max_blocks gets row padding.

    extend_len=[6, 2] at page_size=4 → num_blocks=[2, 1]; max_blocks=2.
    Row 1 (2 tokens, 1 page) is right-padded with page id 0 to width 2.
    """
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],       # req0 pages 0, 1
            [40, 41, 42, 43, 0, 0, 0, 0],   # req1 page 10; tail is unused.
        ],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=6),  # 2 pages
        _FakeReq(table_idx=1, cached_len=0, device_len=2),  # 1 page
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert meta.max_query_len == 6
    assert meta.query_offsets == [0, 6, 8]
    # Row 0: pages [0, 1]. Row 1: page [10] pad-with-0 → [10, 0].
    assert meta.block_table.tolist() == [[0, 1], [10, 0]]
    assert meta.block_table.shape == (2, 2)


def test_gate22f_prepare_metadata_rejects_ragged_with_cached(monkeypatch):
    """Gate 2.2f: any non-zero cached_len in a ragged batch must raise."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # req0: cached=0 extend=4 device=4; req1: cached=2 extend=2 device=4.
    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=4),
        _FakeReq(table_idx=1, cached_len=2, device_len=4),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    with pytest.raises(NotImplementedError) as excinfo:
        backend.prepare_metadata(batch)
    msg = str(excinfo.value)
    assert "cached_len" in msg
    assert "ragged" in msg


def test_gate22f_get_last_indices_ragged(monkeypatch):
    """Gate 2.2f: last-index per request under ragged prefill."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    reqs = [
        _FakeReq(table_idx=0, cached_len=0, device_len=4),
        _FakeReq(table_idx=1, cached_len=0, device_len=2),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)

    # Flat layout: [r0_t0..r0_t3, r1_t0, r1_t1]; last idx per req = [3, 5].
    idx = batch.attn_metadata.get_last_indices(2)
    assert idx.tolist() == [3, 5]
    assert idx.dtype == torch.int32


def _prime_forward_ragged(monkeypatch, query_lens, *, page_size=4,
                          num_heads=4, num_kv_heads=2, head_dim=8,
                          num_pages=32, num_layers=2, layer_id=0):
    """Ragged variant: each request has cached_len==0 and its own extend_len.

    Row ``b`` in page_table gets a disjoint physical-page range so tests can
    trace out_loc.
    """
    import torch as _t

    mod, backend = _make_backend()
    reqs = []
    rows = []
    offset = 0
    row_slots = []
    for b, q_len in enumerate(query_lens):
        nb = (q_len + page_size - 1) // page_size
        row = list(range(offset, offset + nb * page_size))
        # Pad row to a shared width so page_table is rectangular.
        row_slots.append(row)
        rows.append(row)
        offset += nb * page_size
        reqs.append(_FakeReq(table_idx=b, cached_len=0, device_len=q_len))
    max_width = max(len(r) for r in rows)
    padded_rows = [r + [0] * (max_width - len(r)) for r in rows]
    page_table = _t.tensor(padded_rows, dtype=_t.int32)

    kv_cache = _RecordingKVCache(
        num_pages=num_pages, num_kv_heads=num_kv_heads,
        page_size=page_size, head_dim=head_dim,
        num_layers=num_layers, dtype=_t.float32,
    )
    ctx = _FakeCtxFIA(page_table=page_table, page_size=page_size, kv_cache=kv_cache)
    import minisgl.core as core_mod
    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)

    # out_loc = concatenation of per-req real slot ranges.
    ol_pieces = []
    for row, q_len in zip(row_slots, query_lens):
        ol_pieces.extend(row[:q_len])
    out_loc = _t.tensor(ol_pieces, dtype=_t.int32)
    batch = _FakeBatchFIA(padded_reqs=reqs, out_loc=out_loc)

    backend.prepare_metadata(batch)

    total_tokens = sum(query_lens)
    q = _t.randn((total_tokens, num_heads, head_dim), dtype=_t.float32)
    k = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)
    v = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)

    max_q = max(query_lens)
    calls = []
    _install_fake_torch_npu(
        monkeypatch, calls,
        out_shape=(len(query_lens), max_q, num_heads, head_dim),
        out_dtype=_t.float32, out_device=q.device,
    )
    return {
        "backend": backend, "batch": batch, "ctx": ctx, "kv_cache": kv_cache,
        "q": q, "k": k, "v": v, "layer_id": layer_id, "calls": calls,
        "query_lens": query_lens, "max_q": max_q,
        "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "page_size": page_size, "row_slots": row_slots,
    }


def test_gate22f_forward_ragged_query_bsnd_shape(monkeypatch):
    """Gate 2.2f: flat q [sum_q, Hq, D] → packed BSND [B, max_q, Hq, D]."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    query = setup["calls"][0]["query"]
    assert tuple(query.shape) == (2, 4, setup["num_heads"], setup["head_dim"])


def test_gate22f_forward_ragged_pack_unpack_order(monkeypatch):
    """Gate 2.2f: real query rows are packed into positions [0..q_len) per
    batch; padded rows are zero; unpack must recover the flat rows in order.

    We arrange the fake FIA to echo its ``query`` back as the output so
    that unpack correctness reduces to bit-equality between the returned
    flat result and the original ``q``.
    """
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])

    def _echo_fia(query, key, value, **kwargs):
        setup["calls"].append({"query": query, "key": key, "value": value,
                               "kwargs": kwargs})
        # Echo query as attention_out to check pack/unpack round-trip.
        return (query, torch.empty((0,), dtype=torch.float32))
    sys.modules["torch_npu"].npu_fused_infer_attention_score = _echo_fia

    out = setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                                    layer_id=setup["layer_id"],
                                    batch=setup["batch"])
    # Ragged echo returns [B, max_q, H, D] with padding rows zero; unpack
    # must slice each req's real rows back in order.
    assert torch.equal(out, setup["q"].view(setup["q"].shape))
    # Also verify the packed input: batch 0 rows 0..3 = q[0..3]; batch 1
    # rows 0..1 = q[4..5]; batch 1 rows 2..3 = 0.
    packed = setup["calls"][0]["query"]
    assert torch.equal(packed[0, :4], setup["q"][0:4])
    assert torch.equal(packed[1, :2], setup["q"][4:6])
    assert torch.equal(packed[1, 2:], torch.zeros_like(packed[1, 2:]))


def test_gate22f_forward_ragged_mask_shape_and_visibility(monkeypatch):
    """Gate 2.2f: per-batch mask [B, 1, max_q, padded_kv_len]; padded query
    rows are all True (== masked out); real rows are strictly causal.
    """
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    # max_blocks = ceil(4/4) = 1 → padded_kv_len = 1 * 4 = 4.
    assert tuple(mask.shape) == (2, 1, 4, 4)
    # Batch 0 real rows 0..3, KV visible = [1, 2, 3, 4].
    visible_a = (~mask[0, 0, :4]).sum(dim=1).tolist()
    assert visible_a == [1, 2, 3, 4]
    # Batch 1 real rows 0..1, KV visible = [1, 2]. Padding rows 2, 3
    # fully masked.
    visible_b = (~mask[1, 0, :2]).sum(dim=1).tolist()
    assert visible_b == [1, 2]
    assert bool(mask[1, 0, 2:].all().item())


def test_gate22f_forward_ragged_kwargs(monkeypatch):
    """Gate 2.2f: FIA kwargs for ragged prefill carry per-batch seqlens."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    kwargs = setup["calls"][0]["kwargs"]
    assert kwargs["actual_seq_lengths"] == [4, 2]
    assert kwargs["actual_seq_lengths_kv"] == [4, 2]
    assert kwargs["input_layout"] == "BSND"
    # block_table row 0 → page id 0; row 1 → page id 4 (offset by row0's
    # page count = 1 → 1 page * page_size 4 = 4).
    assert kwargs["block_table"].tolist() == [[0], [1]]


def test_gate22f_forward_ragged_store_kv_uses_full_out_loc(monkeypatch):
    """Gate 2.2f: KV store uses the flat ``out_loc`` verbatim; no reorder."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    assert len(setup["kv_cache"].store_kv_calls) == 1
    call = setup["kv_cache"].store_kv_calls[0]
    assert call["k"] is setup["k"]
    assert call["v"] is setup["v"]
    assert call["out_loc"] is setup["batch"].out_loc
    # Total scatter slots equal sum(query_lens); no padding slots injected.
    assert call["out_loc"].numel() == sum(setup["query_lens"])


def test_gate22f_equal_length_b2_still_uses_shared_2d_mask(monkeypatch):
    """Gate 2.2f regression: equal-length B>=1 must still ship a shared 2-D
    causal mask, NOT the per-batch 4-D mask that ragged uses. This keeps the
    Gate 2.2c behaviour intact.
    """
    torch = pytest.importorskip("torch")
    setup = _prime_forward_b2(monkeypatch, query_seq_len=4, kv_seq_len=8, page_size=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    mask = setup["calls"][0]["kwargs"]["atten_mask"]
    assert mask.dim() == 2, (
        f"equal-length B=2 must still use the shared 2-D mask, got shape "
        f"{tuple(mask.shape)}"
    )


def test_gate22f_b1_single_prefill_still_works(monkeypatch):
    """Gate 2.2f regression: B=1 metadata unchanged."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward(monkeypatch, query_seq_len=4, kv_seq_len=4, page_size=4)
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    kwargs = setup["calls"][0]["kwargs"]
    assert kwargs["actual_seq_lengths"] == [4]
    assert kwargs["actual_seq_lengths_kv"] == [4]
    # B=1 equal-length falls through the equal-length shared-mask branch;
    # shape must be [S, padded_kv_len].
    assert kwargs["atten_mask"].dim() == 2


def test_gate22f_forward_flat_query_shape_mismatch_rejected_ragged(monkeypatch):
    """Gate 2.2f: mismatch between metadata sum(query_seq_lens) and flat q
    row count must still raise ValueError."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_ragged(monkeypatch, [4, 2])
    # Truncate q to only 5 rows while metadata claims 4+2=6.
    q_short = setup["q"][:5]
    with pytest.raises(ValueError) as excinfo:
        setup["backend"].forward(q_short, setup["k"][:5], setup["v"][:5],
                                 layer_id=setup["layer_id"], batch=setup["batch"])
    msg = str(excinfo.value)
    assert "flat query" in msg


def _prime_forward_decode_mixed(monkeypatch, kv_lens, *, page_size=4,
                                num_heads=4, num_kv_heads=2, head_dim=8,
                                num_pages=32, num_layers=2, layer_id=0):
    """Pure-decode fixture: each request has ``extend_len==1`` but its own
    ``cached_len`` (and hence its own ``device_len``). Rows in ``page_table``
    are disjoint so tests can trace out_loc / block_table independence.
    """
    import torch as _t

    mod, backend = _make_backend()
    reqs = []
    rows = []
    offset = 0
    for b, kv_len in enumerate(kv_lens):
        nb = (kv_len + page_size - 1) // page_size
        row = list(range(offset, offset + nb * page_size))
        rows.append(row)
        offset += nb * page_size
        reqs.append(_FakeReq(table_idx=b, cached_len=kv_len - 1, device_len=kv_len))
    max_width = max(len(r) for r in rows)
    padded_rows = [r + [0] * (max_width - len(r)) for r in rows]
    page_table = _t.tensor(padded_rows, dtype=_t.int32)

    kv_cache = _RecordingKVCache(
        num_pages=num_pages, num_kv_heads=num_kv_heads,
        page_size=page_size, head_dim=head_dim,
        num_layers=num_layers, dtype=_t.float32,
    )
    ctx = _FakeCtxFIA(page_table=page_table, page_size=page_size, kv_cache=kv_cache)
    import minisgl.core as core_mod
    monkeypatch.setattr(core_mod, "get_global_ctx", lambda: ctx)

    # out_loc = the single new-token slot per request, flat.
    ol_pieces = [rows[b][kv_lens[b] - 1] for b in range(len(kv_lens))]
    out_loc = _t.tensor(ol_pieces, dtype=_t.int32)
    batch = _FakeBatchFIA(padded_reqs=reqs, out_loc=out_loc)

    backend.prepare_metadata(batch)

    total_tokens = len(kv_lens)  # one row per request in decode
    q = _t.randn((total_tokens, num_heads, head_dim), dtype=_t.float32)
    k = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)
    v = _t.randn((total_tokens, num_kv_heads, head_dim), dtype=_t.float32)

    calls = []
    _install_fake_torch_npu(
        monkeypatch, calls,
        out_shape=(len(kv_lens), 1, num_heads, head_dim),
        out_dtype=_t.float32, out_device=q.device,
    )
    return {
        "backend": backend, "batch": batch, "ctx": ctx, "kv_cache": kv_cache,
        "q": q, "k": k, "v": v, "layer_id": layer_id, "calls": calls,
        "kv_lens": kv_lens, "num_heads": num_heads, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim, "page_size": page_size, "rows": rows,
    }


def test_gate22f_prepare_metadata_pure_decode_mixed_cached(monkeypatch):
    """Gate 2.2f: pure-decode batch with different cached_len is accepted;
    metadata reports per-req kv lengths and shared query length 1."""
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    _install_ctx(monkeypatch, page_table, page_size=4)

    # A: cached=4, device=5, extend=1 ; B: cached=2, device=3, extend=1
    reqs = [
        _FakeReq(table_idx=0, cached_len=4, device_len=5),
        _FakeReq(table_idx=1, cached_len=2, device_len=3),
    ]
    batch = _FakeBatch(padded_reqs=reqs)
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata
    assert meta.batch_size == 2
    assert list(meta.query_seq_lens) == [1, 1]
    assert list(meta.kv_seq_lens) == [5, 3]
    assert list(meta.actual_seq_lengths) == [1, 1]
    assert list(meta.actual_seq_lengths_kv) == [5, 3]
    assert meta.max_query_len == 1
    assert list(meta.query_offsets) == [0, 1, 2]
    # Rows independent: row 0 must reference req A's real pages only, row 1 req B's.
    bt = meta.block_table
    assert bt.shape[0] == 2
    # A needs ceil(5/4)=2 blocks; B needs ceil(3/4)=1 block. max_blocks=2.
    assert bt.shape[1] == 2
    # Row 0 (A) real page ids: raw slots 0 and 4 / page_size=4 → pages 0 and 1
    assert bt[0].tolist()[:2] == [0, 1]
    # Row 1 (B) real page id: raw slot 8 / 4 = page 2; second col padded with 0
    assert bt[1, 0].item() == 2
    assert bt[1, 1].item() == 0  # padding column


def test_gate22f_forward_pure_decode_mask_is_none(monkeypatch):
    """Gate 2.2f: pure-decode mixed-cached forward passes atten_mask=None."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_decode_mixed(monkeypatch, [5, 3])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    kwargs = setup["calls"][0]["kwargs"]
    assert kwargs.get("atten_mask") is None
    assert list(kwargs.get("actual_seq_lengths")) == [1, 1]
    assert list(kwargs.get("actual_seq_lengths_kv")) == [5, 3]


def test_gate22f_forward_pure_decode_query_and_output_shape(monkeypatch):
    """Gate 2.2f: [B,Hq,D] flat q → [B,1,Hq,D] BSND; output flattens back."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_decode_mixed(monkeypatch, [5, 3])
    out = setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                                   layer_id=setup["layer_id"], batch=setup["batch"])
    query = setup["calls"][0]["query"]
    assert tuple(query.shape) == (2, 1, setup["num_heads"], setup["head_dim"])
    # Output shape must match the flat q shape the caller supplied.
    assert tuple(out.shape) == tuple(setup["q"].shape)


def test_gate22f_forward_pure_decode_store_kv_uses_full_out_loc(monkeypatch):
    """Gate 2.2f: pure-decode store_kv scatter uses batch.out_loc unchanged."""
    torch = pytest.importorskip("torch")
    setup = _prime_forward_decode_mixed(monkeypatch, [5, 3])
    setup["backend"].forward(setup["q"], setup["k"], setup["v"],
                             layer_id=setup["layer_id"], batch=setup["batch"])
    scatter = setup["kv_cache"].store_kv_calls[0]
    assert scatter["out_loc"].tolist() == setup["batch"].out_loc.tolist()
