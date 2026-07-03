"""Unit tests for Engine's device wiring after Gate 1.3c.

Historical layering:

* Gate 1.1a introduced two engine-local device helpers
  (``_resolve_engine_device`` and ``_bind_engine_local_device``).
* Gate 1.1b consolidated those into
  :func:`minisgl.distributed.runtime.bind_local_device`.
* Gate 1.2b routes ``Engine.__init__``'s Stream creation + binding through the
  shared :mod:`minisgl.utils.device_runtime` (``create_stream`` /
  ``set_stream``) so cuda / npu / cpu all take the same code path.
* Gate 1.3c finishes the migration inside ``Engine.forward_batch``: the raw
  ``torch.cuda.current_stream()`` / ``torch.cuda.Event()`` / ``event.record(...)``
  calls are gone, replaced with the shared ``current_stream`` / ``create_event``
  / ``record_event`` dispatch helpers. ``ForwardOutput.copy_done_event`` no
  longer carries a CUDA-only type annotation.

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
4. Gate 1.3c: ``current_stream`` / ``create_event`` / ``record_event`` are
   imported from ``device_runtime`` and invoked inside ``forward_batch`` with
   ``self.device_type``; the raw ``torch.cuda.current_stream`` /
   ``torch.cuda.Event`` calls are gone from ``forward_batch``; the
   ``ForwardOutput.copy_done_event`` annotation no longer references
   ``torch.cuda.Event`` and the field name / tuple position are preserved.
5. The engine still contains no private device-binding branch (no direct
   ``torch.cuda.set_device`` / ``torch.npu.set_device`` calls, no
   ``import torch_npu`` at module scope).
6. Gate 1.1a's other Ascend-portability guardrails still hold (no hard-coded
   ``backend="nccl"``, ``get_distributed_backend`` still wired in,
   ``get_device_type`` still consulted).
7. The remaining ``TODO(gate-1.2+)`` markers (in ``_sync_get_memory`` and
   ``shutdown``) are still present — those Gates ship separately.

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


def _forward_output_cls() -> ast.ClassDef:
    """Return the ``ast.ClassDef`` for the ``ForwardOutput`` NamedTuple."""
    return next(
        node for node in _engine_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardOutput"
    )


def _init_source() -> str:
    return ast.unparse(_engine_init())


def _method_raw_source(name: str) -> str:
    """Return the raw text of a method (including ``# comments``).

    ``ast.unparse`` drops comments, which would hide any deferred-CUDA TODO
    markers we care about. Slice the source lines by the method's line range
    instead.
    """
    node = _engine_method(name)
    src_lines = _engine_source().splitlines()
    # ast line numbers are 1-based inclusive on both ends.
    return "\n".join(src_lines[node.lineno - 1 : node.end_lineno])


def _forward_batch_source() -> str:
    return _method_raw_source("forward_batch")


def _forward_batch_ast() -> ast.FunctionDef:
    return _engine_method("forward_batch")


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
# Gate 1.2b + 1.3c: device_runtime imports cover both __init__ and forward_batch
# ---------------------------------------------------------------------------


def _device_runtime_imported_names() -> set[str]:
    """Return the set of names imported from ``minisgl.utils.device_runtime``."""
    names: set[str] = set()
    for node in ast.walk(_engine_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "minisgl.utils.device_runtime":
            for alias in node.names:
                names.add(alias.name)
    return names


def test_engine_imports_all_stream_and_event_helpers_from_device_runtime() -> None:
    """__init__ needs create_stream/set_stream; forward_batch needs
    current_stream/create_event/record_event. All five must come from
    ``minisgl.utils.device_runtime``.
    """
    imported = _device_runtime_imported_names()
    for name in (
        "create_stream",
        "set_stream",
        "current_stream",
        "create_event",
        "record_event",
    ):
        assert name in imported, (
            f"{name!r} must be imported from minisgl.utils.device_runtime"
        )


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
# Gate 1.3c: forward_batch routes through device_runtime, not torch.cuda.*
# ---------------------------------------------------------------------------


def test_forward_batch_calls_current_stream_with_device_type() -> None:
    """``current_stream(self.device_type)`` must appear inside forward_batch."""
    fn = _forward_batch_ast()

    matched = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "current_stream":
            continue
        assert len(node.args) == 1 and not node.keywords, (
            "current_stream must take exactly one positional arg (self.device_type)"
        )
        arg = node.args[0]
        assert isinstance(arg, ast.Attribute) and arg.attr == "device_type"
        assert isinstance(arg.value, ast.Name) and arg.value.id == "self"
        matched = True
    assert matched, "forward_batch must call current_stream(self.device_type)"


def test_forward_batch_calls_create_event_with_device_type() -> None:
    """``create_event(self.device_type)`` must appear inside forward_batch."""
    fn = _forward_batch_ast()

    matched = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "create_event":
            continue
        assert len(node.args) == 1 and not node.keywords, (
            "create_event must take exactly one positional arg (self.device_type)"
        )
        arg = node.args[0]
        assert isinstance(arg, ast.Attribute) and arg.attr == "device_type"
        assert isinstance(arg.value, ast.Name) and arg.value.id == "self"
        matched = True
    assert matched, "forward_batch must call create_event(self.device_type)"


def test_forward_batch_calls_record_event_with_device_type_event_and_self_stream() -> None:
    """``record_event(self.device_type, copy_done_event, self.stream)``."""
    fn = _forward_batch_ast()

    matched = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "record_event":
            continue
        assert len(node.args) == 3 and not node.keywords, (
            "record_event must take three positional args (device_type, event, stream)"
        )
        first, second, third = node.args
        # device_type
        assert (
            isinstance(first, ast.Attribute)
            and first.attr == "device_type"
            and isinstance(first.value, ast.Name)
            and first.value.id == "self"
        )
        # event: the local we just created — must be a plain Name (no attr / call)
        assert isinstance(second, ast.Name), (
            "record_event's second arg should be the local event handle "
            "returned by create_event, not an attribute/call chain"
        )
        # stream: must be ``self.stream`` — the same stream __init__ bound.
        assert (
            isinstance(third, ast.Attribute)
            and third.attr == "stream"
            and isinstance(third.value, ast.Name)
            and third.value.id == "self"
        )
        matched = True
    assert matched, (
        "forward_batch must call record_event(self.device_type, <event>, self.stream)"
    )


def test_forward_batch_no_longer_calls_torch_cuda_current_stream() -> None:
    src = _forward_batch_source()
    assert "torch.cuda.current_stream" not in src, (
        "forward_batch must route through device_runtime.current_stream, "
        "not torch.cuda.current_stream directly"
    )


def test_forward_batch_no_longer_calls_torch_cuda_event() -> None:
    src = _forward_batch_source()
    assert "torch.cuda.Event" not in src, (
        "forward_batch must route through device_runtime.create_event/"
        "record_event, not torch.cuda.Event directly"
    )


def test_forward_batch_no_longer_calls_event_record_directly() -> None:
    """The old ``copy_done_event.record(self.stream)`` pattern is gone.

    Recording must go through ``record_event(self.device_type, ...)`` so the
    NPU / CPU branches can dispatch correctly. Any attribute call whose method
    name is ``record`` on a local Name (i.e. ``copy_done_event.record(...)``)
    is forbidden inside forward_batch.
    """
    fn = _forward_batch_ast()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "record":
            continue
        # A local ``foo.record(...)`` call — flag it.
        assert not isinstance(func.value, ast.Name), (
            f"forward_batch must not call ``{func.value.id}.record(...)`` "
            "directly; use device_runtime.record_event instead"
        )


def test_forward_batch_has_no_gate_1_2_todo_markers() -> None:
    """Gate 1.3c completes the deferred migration inside forward_batch, so its
    ``TODO(gate-1.2+)`` markers must be gone. Other methods keep theirs.
    """
    src = _forward_batch_source()
    assert "TODO(gate-1.2+)" not in src, (
        "forward_batch's deferred CUDA calls were ported in Gate 1.3c — the "
        "TODO(gate-1.2+) marker(s) inside forward_batch must be removed"
    )


# ---------------------------------------------------------------------------
# Gate 1.3c: ForwardOutput type annotation is now device-agnostic
# ---------------------------------------------------------------------------


def test_forward_output_still_has_copy_done_event_field() -> None:
    """The public field name and its position in the NamedTuple must not change.

    ``scheduler._process_last_data`` unpacks ``ForwardOutput`` positionally as
    ``(_, next_tokens_cpu, copy_done)`` — reordering the fields would silently
    break the scheduler. Assert both the field name and the tuple position.
    """
    cls = _forward_output_cls()
    field_names = [
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert field_names == [
        "next_tokens_gpu",
        "next_tokens_cpu",
        "copy_done_event",
    ], f"ForwardOutput field order changed: {field_names!r}"


def test_forward_output_annotation_no_longer_references_torch_cuda_event() -> None:
    """The ``copy_done_event`` annotation must be device-agnostic (no ``torch.cuda.Event``)."""
    cls = _forward_output_cls()
    ann = next(
        stmt.annotation
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == "copy_done_event"
    )
    # The whole annotation subtree must contain no ``torch.cuda.Event`` chain.
    ann_src = ast.unparse(ann)
    assert "torch.cuda.Event" not in ann_src, (
        f"copy_done_event annotation still references torch.cuda.Event: {ann_src!r}"
    )


def test_engine_module_imports_any_from_typing() -> None:
    """The device-agnostic ``Any | None`` annotation requires ``Any`` in scope."""
    imported: set[str] = set()
    for node in ast.walk(_engine_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                imported.add(alias.name)
    assert "Any" in imported, "engine.py must import Any from typing"


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
    """Memory helpers + graph capture still carry ``TODO(gate-1.2+)`` markers."""
    src = _engine_source()
    assert "TODO(gate-1.2+)" in src
