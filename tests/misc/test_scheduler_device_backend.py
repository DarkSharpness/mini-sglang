"""Gate 1.11g — Scheduler stream lifecycle must route through device_runtime.

Structural + behavioural checks on ``minisgl.scheduler.scheduler``:

1. The module must not contain any executable ``torch.cuda.Stream``,
   ``torch.cuda.stream``, ``torch.cuda.set_stream``, ``torch.cuda.current_stream``,
   or ``torch.cuda.synchronize`` reference. All stream/synchronize primitives
   must be sourced from ``minisgl.utils.device_runtime``.
2. The module must import ``create_stream``, ``set_stream``, ``current_stream``,
   ``stream_context``, and ``synchronize_device`` from ``minisgl.utils.device_runtime``.
3. ``Scheduler.__init__`` on NPU never accesses ``torch.cuda.Stream`` or
   ``torch.cuda.set_stream`` — we booby-trap them and drive init through a stub
   Engine to prove the NPU branch is taken.
4. ``Scheduler.shutdown`` calls ``synchronize_device`` with the engine's
   ``device_type``, not raw ``torch.cuda.synchronize``.
5. The CUDA branch remains structurally intact (device_runtime still dispatches
   to ``torch.cuda.Stream`` etc. when ``device_type=="cuda"``).
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEDULER_PATH = _REPO_ROOT / "python" / "minisgl" / "scheduler" / "scheduler.py"


def _purge_minisgl_from_sys_modules() -> None:
    """Some earlier tests (test_device_runtime) install stub ``minisgl.utils``
    package objects into ``sys.modules`` at import time to avoid loading heavy
    optional deps. Those stubs shadow the real package for later imports in
    the same session, breaking ``from minisgl.utils import cached_load_hf_config``
    inside ``minisgl.engine.config``. We evict every ``minisgl*`` entry so the
    behavioural tests below get a clean re-import.
    """
    for name in list(sys.modules):
        if name == "minisgl" or name.startswith("minisgl."):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# 1. Structural: scheduler.py must not contain any raw torch.cuda stream call.
# ---------------------------------------------------------------------------


_FORBIDDEN_PATTERNS = [
    r"torch\.cuda\.Stream\b",
    r"torch\.cuda\.stream\b",
    r"torch\.cuda\.set_stream\b",
    r"torch\.cuda\.current_stream\b",
    r"torch\.cuda\.synchronize\b",
]


def test_scheduler_source_has_no_raw_torch_cuda_stream_calls() -> None:
    src = _SCHEDULER_PATH.read_text()
    # Strip comments so prose about torch.cuda.* is not counted.
    code_only = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for pattern in _FORBIDDEN_PATTERNS:
        assert not re.search(pattern, code_only), (
            f"scheduler.py must not contain executable {pattern!r} — "
            "route through minisgl.utils.device_runtime instead"
        )


def test_scheduler_imports_device_runtime_helpers() -> None:
    tree = ast.parse(_SCHEDULER_PATH.read_text())
    imported: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )
    dr_imports = imported.get("minisgl.utils.device_runtime", set())
    for name in ("create_stream", "set_stream", "current_stream",
                 "stream_context", "synchronize_device"):
        assert name in dr_imports, (
            f"scheduler.py must import {name!r} from "
            f"minisgl.utils.device_runtime; got {dr_imports}"
        )


# ---------------------------------------------------------------------------
# 2. Behavioural: Scheduler.__init__ on NPU never touches torch.cuda stream API.
# ---------------------------------------------------------------------------


class _StreamCtx:
    """Fake StreamContext returned by torch.{cuda,npu}.stream(s)."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def __enter__(self) -> "_StreamCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _ExplodingCudaStream:
    """Sentinel for torch.cuda.Stream — any call blows up."""

    def __call__(self, *args, **kwargs) -> None:
        raise AssertionError(
            "Scheduler.__init__ must NOT call torch.cuda.Stream on a non-CUDA host"
        )


def _make_stub_engine(monkeypatch: pytest.MonkeyPatch, device_type: str) -> Any:
    """Build a minimal object that quacks like ``minisgl.engine.Engine`` for
    Scheduler.__init__. Only the fields Scheduler actually touches are populated.
    """
    import torch

    engine = MagicMock()
    engine.device_type = device_type
    engine.device = torch.device("cpu")  # only stored, never used for CUDA ops here
    # Use a plain sentinel object as the engine's stream — dispatch tests don't
    # need a real backend stream, only that stream_context receives THIS object.
    engine.stream = object()
    engine.num_pages = 16
    engine.page_table = torch.zeros((4, 16), dtype=torch.int32)
    engine.tp_cpu_group = None

    # Scheduler's constructor calls `Engine(config)` which resolves via
    # `from minisgl.engine import Engine`. Patch that lookup to return a factory
    # that yields our stub, ignoring the SchedulerConfig arg.
    def _fake_engine_ctor(_config):
        return engine

    from minisgl import engine as engine_pkg

    monkeypatch.setattr(engine_pkg, "Engine", _fake_engine_ctor, raising=True)
    return engine


def _make_stub_config() -> Any:
    """Minimal SchedulerConfig stand-in — Scheduler only reads a few fields
    after Engine construction returns."""
    cfg = MagicMock()
    cfg.max_running_req = 4
    cfg.page_size = 16
    cfg.cache_type = "radix"
    cfg.max_extend_tokens = 8192
    cfg.model_path = "unused"
    cfg.offline_mode = True
    cfg.tp_info = MagicMock(rank=0, size=1, is_primary=lambda: True)
    return cfg


def test_npu_scheduler_init_never_touches_torch_cuda_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _purge_minisgl_from_sys_modules()

    # Patch remaining collaborators that Scheduler.__init__ pulls in.
    import minisgl.utils.device_runtime as dr

    calls: List[Tuple[str, Any]] = []

    def _fake_create_stream(dt: str) -> Any:
        calls.append(("create_stream", dt))
        return object()

    def _fake_set_stream(dt: str, s: Any) -> None:
        calls.append(("set_stream", dt))

    def _fake_stream_context(dt: str, s: Any) -> Any:
        calls.append(("stream_context", dt))
        return _StreamCtx(s)

    monkeypatch.setattr(dr, "create_stream", _fake_create_stream)
    monkeypatch.setattr(dr, "set_stream", _fake_set_stream)
    monkeypatch.setattr(dr, "stream_context", _fake_stream_context)
    # Also patch the names as imported into scheduler.py's module namespace.
    import minisgl.scheduler.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "create_stream", _fake_create_stream)
    monkeypatch.setattr(sched_mod, "set_stream", _fake_set_stream)
    monkeypatch.setattr(sched_mod, "stream_context", _fake_stream_context)

    # Booby-trap the raw torch.cuda stream API so any accidental call fails hard.
    monkeypatch.setattr(torch.cuda, "Stream", _ExplodingCudaStream(), raising=False)

    def _explode_set_stream(_s: Any) -> None:
        raise AssertionError("Scheduler.__init__ must NOT call torch.cuda.set_stream")

    def _explode_current_stream() -> None:
        raise AssertionError(
            "Scheduler.__init__ must NOT call torch.cuda.current_stream"
        )

    monkeypatch.setattr(torch.cuda, "set_stream", _explode_set_stream, raising=False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", _explode_current_stream, raising=False
    )

    # Skip heavy sub-managers by pre-patching their constructors.
    monkeypatch.setattr(sched_mod, "TableManager", MagicMock())
    monkeypatch.setattr(sched_mod, "CacheManager", MagicMock())
    monkeypatch.setattr(sched_mod, "DecodeManager", MagicMock())
    monkeypatch.setattr(sched_mod, "PrefillManager", MagicMock())
    monkeypatch.setattr(sched_mod, "load_tokenizer", lambda _p: MagicMock(eos_token_id=0))

    # Build the stub engine + config.
    _make_stub_engine(monkeypatch, device_type="npu")
    cfg = _make_stub_config()

    scheduler = sched_mod.Scheduler(cfg)

    # Exactly one create_stream, one stream_context, one set_stream — all with
    # device_type == "npu". No touch of raw torch.cuda stream API.
    call_names = [c[0] for c in calls]
    assert call_names == ["create_stream", "stream_context", "set_stream"], call_names
    for name, dt in calls:
        assert dt == "npu", f"{name} routed with device_type={dt} instead of npu"
    assert scheduler.device_type == "npu"


def test_npu_scheduler_shutdown_routes_through_device_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _purge_minisgl_from_sys_modules()

    import minisgl.scheduler.scheduler as sched_mod
    import minisgl.utils.device_runtime as dr

    sync_calls: List[str] = []

    def _fake_synchronize_device(dt: str) -> None:
        sync_calls.append(dt)

    monkeypatch.setattr(dr, "synchronize_device", _fake_synchronize_device)
    monkeypatch.setattr(sched_mod, "synchronize_device", _fake_synchronize_device)

    def _explode_cuda_synchronize(_d=None) -> None:
        raise AssertionError(
            "Scheduler.shutdown must NOT call torch.cuda.synchronize on a non-CUDA host"
        )

    monkeypatch.setattr(torch.cuda, "synchronize", _explode_cuda_synchronize, raising=False)

    # Build a Scheduler skeleton — bypass __init__ so we only exercise shutdown.
    scheduler = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    scheduler.device_type = "npu"
    scheduler.engine = MagicMock()
    # Stub sync_all_ranks so we don't need a real ProcessGroup.
    scheduler.sync_all_ranks = lambda: None  # type: ignore[method-assign]

    scheduler.shutdown()

    assert sync_calls == ["npu"], sync_calls
    scheduler.engine.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# 3. CUDA branch preserved — device_runtime dispatches to torch.cuda.*.
# ---------------------------------------------------------------------------


def test_cuda_branch_still_reaches_torch_cuda_stream_via_device_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When device_type=='cuda', device_runtime dispatches to torch.cuda.*
    exactly like before. This proves the refactor did not amputate the CUDA
    path — it just moved it behind the dispatch helper."""
    torch = pytest.importorskip("torch")
    _purge_minisgl_from_sys_modules()

    import minisgl.utils.device_runtime as dr

    got: List[str] = []
    orig_cuda_stream_cls = torch.cuda.Stream
    orig_cuda_set_stream = torch.cuda.set_stream
    orig_cuda_synchronize = torch.cuda.synchronize

    def _fake_stream_ctor(*_a, **_kw) -> Any:
        got.append("cuda.Stream")
        return object()

    def _fake_set_stream(_s: Any) -> None:
        got.append("cuda.set_stream")

    def _fake_synchronize(*_a, **_kw) -> None:
        got.append("cuda.synchronize")

    class _FakeStreamCtx:
        def __init__(self, s: Any) -> None:
            got.append("cuda.stream")

        def __enter__(self) -> "_FakeStreamCtx":
            return self

        def __exit__(self, *_a) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "Stream", _fake_stream_ctor, raising=False)
    monkeypatch.setattr(torch.cuda, "set_stream", _fake_set_stream, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", _fake_synchronize, raising=False)
    monkeypatch.setattr(torch.cuda, "stream", _FakeStreamCtx, raising=False)

    try:
        s = dr.create_stream("cuda")
        dr.set_stream("cuda", s)
        with dr.stream_context("cuda", s):
            pass
        dr.synchronize_device("cuda")
    finally:
        torch.cuda.Stream = orig_cuda_stream_cls  # type: ignore[assignment]
        torch.cuda.set_stream = orig_cuda_set_stream  # type: ignore[assignment]
        torch.cuda.synchronize = orig_cuda_synchronize  # type: ignore[assignment]

    assert got == ["cuda.Stream", "cuda.set_stream", "cuda.stream", "cuda.synchronize"]
