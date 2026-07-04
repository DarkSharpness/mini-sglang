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


def test_prepare_metadata_returns_none():
    mod = _load_ascend_fia_module()
    backend = mod.AscendFIABackend(config=None)
    assert backend.prepare_metadata(batch=None) is None


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
