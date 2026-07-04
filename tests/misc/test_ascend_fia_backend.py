"""Gate 1.8a: hermetic tests for the AscendFIABackend skeleton and its wiring.

The class module (``minisgl.attention.ascend_fia``) is torch-free at import
time, so we can exercise real Python semantics — instantiation, method calls,
and registry state — without pulling in the CUDA / Ascend runtime.

The engine wiring test uses ``ast``-inspection because
``minisgl.engine.engine`` transitively imports torch + huggingface + kernels.

What's checked here:

1. ``AscendFIABackend`` is instantiable (all abstract methods overridden).
2. All five ``BaseAttnBackend`` interface methods exist on the class.
3. ``init_capture_graph`` / ``prepare_for_capture`` / ``prepare_for_replay``
   are callable no-ops (return ``None``).
4. ``prepare_metadata`` returns ``None``.
5. ``forward`` raises the exact ``NotImplementedError`` mandated by the spec.
6. ``SUPPORTED_ATTENTION_BACKENDS`` contains ``"npu_fia"``.
7. Registering ``"npu_fia"`` does not import ``ascend_fia`` until the factory
   is actually invoked (lazy import).
8. ``_adjust_config`` maps ``device_type == "npu"`` + ``auto`` → ``"npu_fia"``.
9. ``_adjust_config`` keeps the existing SM100 / SM90 / other selection on
   CUDA/CPU auto.
10. Explicit ``fi`` / ``fa`` / ``trtllm`` / ``npu_fia`` are not overridden.
11. ``ascend_fia`` module has no top-level ``torch_npu`` import.
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


# --------------------------------- 5: forward NotImplementedError -----------


def test_forward_raises_gate_1_8b_deferred_error():
    mod = _load_ascend_fia_module()
    backend = mod.AscendFIABackend(config=None)
    with pytest.raises(NotImplementedError) as excinfo:
        backend.forward(q=None, k=None, v=None, layer_id=0, batch=None)
    assert "Ascend FIA forward is not implemented until Gate 1.8b" in str(excinfo.value)


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


def test_ascend_fia_module_never_calls_or_lazy_imports_torch_npu():
    """Belt-and-suspenders check — even lazy imports inside functions are
    forbidden for Gate 1.8a. The FIA op wiring lands separately in 1.8b.

    We walk the AST so docstring prose that mentions ``torch_npu`` for
    documentation purposes is allowed; only actual ``Import`` / ``ImportFrom``
    / ``Attribute`` / ``Name`` references are rejected.
    """
    tree = ast.parse(_ascend_fia_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch_npu"), (
                    f"lazy `import {alias.name}` forbidden in ascend_fia.py "
                    "for Gate 1.8a"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch_npu"), (
                f"lazy `from {node.module} import ...` forbidden in ascend_fia.py "
                "for Gate 1.8a"
            )
        elif isinstance(node, ast.Attribute):
            # `torch_npu.foo` or `torch_npu.foo.bar` — chase the root name.
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                assert root.id != "torch_npu", (
                    "ascend_fia.py must not reference torch_npu.* for Gate 1.8a"
                )
        elif isinstance(node, ast.Name):
            assert node.id != "torch_npu", (
                "ascend_fia.py must not reference `torch_npu` for Gate 1.8a"
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


def test_get_last_indices_bs_not_one_raises(monkeypatch):
    torch = pytest.importorskip("torch")
    mod, backend = _make_backend()
    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    _install_ctx(monkeypatch, page_table, page_size=4)

    req = _FakeReq(table_idx=0, cached_len=0, device_len=4)
    batch = _FakeBatch(padded_reqs=[req])
    backend.prepare_metadata(batch)

    with pytest.raises(NotImplementedError) as excinfo:
        batch.attn_metadata.get_last_indices(2)
    assert "batch size 1 only" in str(excinfo.value)


# --------------------------------- multi-request rejection ---------------


def test_prepare_metadata_multi_request_raises(monkeypatch):
    """`len(padded_reqs) > 1` must fail loudly until the TND path lands."""
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
    with pytest.raises(NotImplementedError) as excinfo:
        backend.prepare_metadata(batch)
    assert "batch size 1 only" in str(excinfo.value)


# --------------------------------- Gate 1.8a invariants (still hold) -----


def test_gate18c_forward_still_raises_gate18b_error():
    """Even with metadata implemented, forward() stays a NotImplementedError
    until the FIA op wiring lands."""
    mod = _load_ascend_fia_module()
    backend = mod.AscendFIABackend(config=None)
    with pytest.raises(NotImplementedError) as excinfo:
        backend.forward(q=None, k=None, v=None, layer_id=0, batch=None)
    assert "Ascend FIA forward is not implemented until Gate 1.8b" in str(excinfo.value)


def test_gate18c_module_still_never_references_torch_npu():
    """Re-assert the Gate 1.8a top-level + lazy torch_npu ban, because
    Gate 1.8c added new function bodies that could regress it."""
    tree = ast.parse(_ascend_fia_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch_npu"), (
                    f"`import {alias.name}` forbidden in ascend_fia.py until "
                    "the FIA op wiring Gate"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch_npu"), (
                f"`from {node.module} import ...` forbidden in ascend_fia.py"
            )
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                assert root.id != "torch_npu", (
                    "ascend_fia.py must not reference torch_npu.*"
                )
        elif isinstance(node, ast.Name):
            assert node.id != "torch_npu", (
                "ascend_fia.py must not reference `torch_npu`"
            )
