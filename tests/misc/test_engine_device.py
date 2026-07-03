"""Unit tests for Engine's device wiring after Gate 1.1b.

Gate 1.1a introduced two engine-local helpers (``_resolve_engine_device`` and
``_bind_engine_local_device``) to de-CUDA-ify ``Engine.__init__``. Gate 1.1b
consolidates that logic into the shared
:func:`minisgl.distributed.runtime.bind_local_device` primitive, so both the
runtime bootstrap and the engine bootstrap go through **one** implementation.

These tests are purely source-level: importing ``minisgl.engine.engine`` on a
stock macOS box would pull in torch, transformers, huggingface_hub and the
attention kernels, none of which are available. Instead we ``ast``-parse the
engine module and assert:

1. The Gate 1.1a helpers are gone.
2. The shared ``bind_local_device`` is imported and called from ``Engine.__init__``.
3. The engine no longer contains any private device-binding branch (no direct
   ``torch.cuda.set_device`` / ``torch.npu.set_device`` calls, no
   ``import torch_npu`` at module scope).
4. Gate 1.1a's other Ascend-portability guardrails still hold (no hard-coded
   ``backend="nccl"``, ``get_distributed_backend`` still wired in,
   ``get_device_type`` still consulted).

Coverage of ``bind_local_device`` itself (cuda / npu / cpu behaviour, npu
import failure, CPU no-op) lives in :mod:`tests.misc.test_distributed_runtime`
where the runtime module is loaded hermetically.
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


# ---------------------------------------------------------------------------
# The two Gate 1.1a helpers have been removed
# ---------------------------------------------------------------------------


def test_engine_no_longer_defines_resolve_engine_device() -> None:
    for node in _engine_tree().body:
        if isinstance(node, ast.FunctionDef):
            assert node.name != "_resolve_engine_device", (
                "_resolve_engine_device must be removed in Gate 1.1b — "
                "engine should reuse bind_local_device instead"
            )


def test_engine_no_longer_defines_bind_engine_local_device() -> None:
    for node in _engine_tree().body:
        if isinstance(node, ast.FunctionDef):
            assert node.name != "_bind_engine_local_device", (
                "_bind_engine_local_device must be removed in Gate 1.1b — "
                "engine should reuse bind_local_device instead"
            )


def test_engine_has_no_private_device_helpers() -> None:
    """Neither helper name may appear anywhere in the engine source."""
    src = _engine_source()
    assert "_resolve_engine_device" not in src
    assert "_bind_engine_local_device" not in src


# ---------------------------------------------------------------------------
# Engine imports & calls the shared primitive
# ---------------------------------------------------------------------------


def test_engine_imports_bind_local_device_from_runtime() -> None:
    src = _engine_source()
    assert "from minisgl.distributed.runtime import bind_local_device" in src


def test_engine_calls_bind_local_device_from_runtime() -> None:
    """The shared primitive must be invoked in Engine.__init__."""
    src = _engine_source()
    assert "bind_local_device(" in src

    # Locate Engine.__init__ and confirm the call happens inside it (not just
    # in a docstring or comment).
    tree = _engine_tree()
    engine_cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    init_fn = next(
        node for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = [
        n for n in ast.walk(init_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "bind_local_device"
    ]
    assert calls, "Engine.__init__ must call bind_local_device()"


# ---------------------------------------------------------------------------
# No leftover private device branch inside engine.py
# ---------------------------------------------------------------------------


def test_engine_does_not_call_torch_cuda_set_device_directly() -> None:
    """The CUDA set_device call now lives inside bind_local_device, not engine.py."""
    src = _engine_source()
    assert "torch.cuda.set_device" not in src


def test_engine_does_not_call_torch_npu_set_device_directly() -> None:
    """The NPU set_device call now lives inside bind_local_device, not engine.py."""
    src = _engine_source()
    assert "torch.npu.set_device" not in src


def test_engine_does_not_import_torch_npu_at_any_scope() -> None:
    """``import torch_npu`` must not appear anywhere in engine.py post-Gate-1.1b."""
    src = _engine_source()
    assert "torch_npu" not in src, (
        "engine.py must not reference torch_npu — the dynamic import lives "
        "inside minisgl.distributed.runtime.bind_local_device"
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


def test_engine_source_marks_deferred_cuda_calls() -> None:
    src = _engine_source()
    assert "TODO(gate-1.2+)" in src
    # The deferred stream/event/graph sites still exist verbatim.
    assert "torch.cuda.Stream" in src
    assert "torch.cuda.Event" in src
