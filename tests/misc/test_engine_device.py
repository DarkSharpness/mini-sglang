"""Unit tests for Engine's device wiring after Gate 1.2b.

Historical layering:

* Gate 1.1a introduced two engine-local device helpers
  (``_resolve_engine_device`` and ``_bind_engine_local_device``).
* Gate 1.1b consolidated those into
  :func:`minisgl.distributed.runtime.bind_local_device`.
* Gate 1.2b routes ``Engine.__init__``'s Stream creation + binding through the
  shared :mod:`minisgl.utils.device_runtime` (``create_stream`` /
  ``set_stream``) so cuda / npu / cpu all take the same code path.

These tests are purely source-level: importing ``minisgl.engine.engine`` on a
stock macOS box would pull in torch, transformers, huggingface_hub and the
attention kernels, none of which are available. Instead we ``ast``-parse the
engine module and assert the required guardrails.

What's checked here:

1. Gate 1.1a helpers stay removed.
2. Gate 1.1b's shared ``bind_local_device`` is still imported and called from
   ``Engine.__init__``.
3. Gate 1.2b: ``create_stream`` / ``set_stream`` from ``device_runtime`` are
   imported and invoked in ``Engine.__init__``; the raw ``torch.cuda.Stream``
   / ``torch.cuda.set_stream`` calls are gone from ``__init__``.
4. ``forward_batch``'s ``torch.cuda.current_stream()`` and
   ``torch.cuda.Event()`` are *deliberately* preserved with their
   ``TODO(gate-1.2+)`` markers — a later Gate ports them.
5. The engine still contains no private device-binding branch (no direct
   ``torch.cuda.set_device`` / ``torch.npu.set_device`` calls, no
   ``import torch_npu`` at module scope).
6. Gate 1.1a's other Ascend-portability guardrails still hold (no hard-coded
   ``backend="nccl"``, ``get_distributed_backend`` still wired in,
   ``get_device_type`` still consulted).

Coverage of the primitives themselves lives in dedicated hermetic suites
(``test_distributed_runtime`` for ``bind_local_device``,
``test_device_runtime`` for ``create_stream`` / ``set_stream`` / ...).
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_PATH = _REPO_ROOT / "python" / "minisgl" / "engine" / "engine.py"


def _engine_source() -> str:
    return _ENGINE_PATH.read_text()


def _engine_tree() -> ast.Module:
    return ast.parse(_engine_source())


def _engine_init() -> ast.FunctionDef:
    """Return the ``ast.FunctionDef`` of ``Engine.__init__``."""
    engine_cls = next(
        node for node in _engine_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    return next(
        node for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def _engine_method(name: str) -> ast.FunctionDef:
    engine_cls = next(
        node for node in _engine_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    return next(
        node for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _init_source() -> str:
    return ast.unparse(_engine_init())


def _method_raw_source(name: str) -> str:
    """Return the raw text of a method (including ``# comments``).

    ``ast.unparse`` drops comments, which would hide the deferred-CUDA TODO
    markers we care about. Slice the source lines by the method's line range
    instead.
    """
    node = _engine_method(name)
    src_lines = _engine_source().splitlines()
    # ast line numbers are 1-based inclusive on both ends.
    return "\n".join(src_lines[node.lineno - 1 : node.end_lineno])


def _forward_batch_source() -> str:
    return _method_raw_source("forward_batch")


# ---------------------------------------------------------------------------
# The two Gate 1.1a helpers stay removed
# ---------------------------------------------------------------------------


def test_engine_no_longer_defines_resolve_engine_device() -> None:
    for node in _engine_tree().body:
        if isinstance(node, ast.FunctionDef):
            assert node.name != "_resolve_engine_device"


def test_engine_no_longer_defines_bind_engine_local_device() -> None:
    for node in _engine_tree().body:
        if isinstance(node, ast.FunctionDef):
            assert node.name != "_bind_engine_local_device"


def test_engine_has_no_private_device_helpers() -> None:
    src = _engine_source()
    assert "_resolve_engine_device" not in src
    assert "_bind_engine_local_device" not in src


# ---------------------------------------------------------------------------
# Gate 1.1b: shared bind_local_device still wired in
# ---------------------------------------------------------------------------


def test_engine_imports_bind_local_device_from_runtime() -> None:
    src = _engine_source()
    assert "from minisgl.distributed.runtime import bind_local_device" in src


def test_engine_init_calls_bind_local_device() -> None:
    init_fn = _engine_init()
    calls = [
        n for n in ast.walk(init_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "bind_local_device"
    ]
    assert calls, "Engine.__init__ must call bind_local_device()"


# ---------------------------------------------------------------------------
# Gate 1.2b: Stream dispatch through device_runtime
# ---------------------------------------------------------------------------


def test_engine_imports_stream_helpers_from_device_runtime() -> None:
    """Engine.__init__ must source ``create_stream`` + ``set_stream`` from device_runtime."""
    src = _engine_source()
    assert "from minisgl.utils.device_runtime import" in src
    # Both names must be imported. Robust against either "a, b" or "b, a" order.
    import_line = next(
        line for line in src.splitlines()
        if line.startswith("from minisgl.utils.device_runtime import")
    )
    assert "create_stream" in import_line
    assert "set_stream" in import_line


def test_engine_init_calls_create_stream_with_device_type() -> None:
    """``self.stream = create_stream(self.device_type)`` must appear in __init__."""
    init_fn = _engine_init()

    matched = False
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "create_stream":
            continue
        # Single positional arg must be ``self.device_type``.
        assert len(node.args) == 1 and not node.keywords, (
            "create_stream must be called with exactly one positional arg (self.device_type)"
        )
        arg = node.args[0]
        assert isinstance(arg, ast.Attribute) and arg.attr == "device_type"
        assert isinstance(arg.value, ast.Name) and arg.value.id == "self"
        matched = True
    assert matched, "Engine.__init__ must call create_stream(self.device_type)"


def test_engine_init_calls_set_stream_with_device_type_and_self_stream() -> None:
    """``set_stream(self.device_type, self.stream)`` must appear in __init__."""
    init_fn = _engine_init()

    matched = False
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "set_stream":
            continue
        assert len(node.args) == 2 and not node.keywords, (
            "set_stream must be called with two positional args"
        )
        first, second = node.args
        assert (
            isinstance(first, ast.Attribute)
            and first.attr == "device_type"
            and isinstance(first.value, ast.Name)
            and first.value.id == "self"
        )
        assert (
            isinstance(second, ast.Attribute)
            and second.attr == "stream"
            and isinstance(second.value, ast.Name)
            and second.value.id == "self"
        )
        matched = True
    assert matched, "Engine.__init__ must call set_stream(self.device_type, self.stream)"


def test_engine_init_no_longer_calls_torch_cuda_stream_directly() -> None:
    """``torch.cuda.Stream(`` must not appear inside Engine.__init__ anymore."""
    src = _init_source()
    assert "torch.cuda.Stream" not in src, (
        "Engine.__init__ must go through device_runtime.create_stream, "
        "not torch.cuda.Stream directly"
    )


def test_engine_init_no_longer_calls_torch_cuda_set_stream_directly() -> None:
    """``torch.cuda.set_stream(`` must not appear inside Engine.__init__ anymore."""
    src = _init_source()
    assert "torch.cuda.set_stream" not in src, (
        "Engine.__init__ must go through device_runtime.set_stream, "
        "not torch.cuda.set_stream directly"
    )


# ---------------------------------------------------------------------------
# Deliberate leftovers: forward_batch still uses CUDA current_stream + Event
# ---------------------------------------------------------------------------


def test_forward_batch_still_uses_torch_cuda_current_stream() -> None:
    """Gate 1.2b defers ``current_stream()`` to a later Gate — must stay put."""
    src = _forward_batch_source()
    assert "torch.cuda.current_stream" in src


def test_forward_batch_still_uses_torch_cuda_event() -> None:
    """Gate 1.2b defers ``Event`` to a later Gate — must stay put."""
    src = _forward_batch_source()
    assert "torch.cuda.Event" in src


def test_forward_batch_deferred_cuda_calls_are_annotated_with_todos() -> None:
    """Each deferred CUDA call in ``forward_batch`` carries a ``TODO(gate-`` marker."""
    src = _forward_batch_source()
    assert "TODO(gate-" in src


# ---------------------------------------------------------------------------
# No leftover private device branch inside engine.py
# ---------------------------------------------------------------------------


def test_engine_does_not_call_torch_cuda_set_device_directly() -> None:
    src = _engine_source()
    assert "torch.cuda.set_device" not in src


def test_engine_does_not_call_torch_npu_set_device_directly() -> None:
    src = _engine_source()
    assert "torch.npu.set_device" not in src


def test_engine_does_not_import_torch_npu_at_any_scope() -> None:
    src = _engine_source()
    assert "torch_npu" not in src, (
        "engine.py must not reference torch_npu — the dynamic import lives "
        "inside minisgl.distributed.runtime.bind_local_device and "
        "minisgl.utils.device_runtime"
    )


# ---------------------------------------------------------------------------
# Gate 1.1a guardrails still hold
# ---------------------------------------------------------------------------


def test_engine_source_has_no_hardcoded_nccl_backend() -> None:
    src = _engine_source()
    assert 'backend="nccl"' not in src
    assert "backend='nccl'" not in src


def test_engine_source_uses_get_distributed_backend() -> None:
    src = _engine_source()
    assert "from minisgl.distributed.backend import get_distributed_backend" in src
    assert "get_distributed_backend(self.device_type)" in src


def test_engine_source_uses_unified_device_type() -> None:
    src = _engine_source()
    assert "from minisgl.utils.device import" in src
    assert "get_device_type" in src


def test_engine_source_still_marks_remaining_deferred_cuda_calls() -> None:
    """The Event/memory/graph deferrals still carry ``TODO(gate-1.2+)`` markers."""
    src = _engine_source()
    assert "TODO(gate-1.2+)" in src
