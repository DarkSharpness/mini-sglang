"""Unit tests for :mod:`minisgl.utils.device_runtime`.

These tests never require a real CUDA / NPU install. They inject fake
``torch`` and ``torch_npu`` modules into ``sys.modules`` and swap the
``builtins.__import__`` hook to control ``import torch_npu`` failures.

The module under test is loaded via ``importlib.util`` so we bypass
``minisgl.utils.__init__`` (which pulls in heavyweight optional deps like
``transformers`` and ``huggingface_hub``). This mirrors the pattern used by
:mod:`tests.misc.test_device` and :mod:`tests.misc.test_distributed_runtime`.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY_ROOT = _REPO_ROOT / "python"
_DEVICE_PATH = _PY_ROOT / "minisgl" / "utils" / "device.py"
_DEVICE_RUNTIME_PATH = _PY_ROOT / "minisgl" / "utils" / "device_runtime.py"


def _install_package_stub(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


def _load_isolated(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot build spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register minimal package stubs so ``from .device import ...`` inside
# device_runtime.py resolves without executing the real utils/__init__.py.
_install_package_stub("minisgl", _PY_ROOT / "minisgl")
_install_package_stub("minisgl.utils", _PY_ROOT / "minisgl" / "utils")

device = _load_isolated("minisgl.utils.device", _DEVICE_PATH)
dr = _load_isolated("minisgl.utils.device_runtime", _DEVICE_RUNTIME_PATH)


# ---------------------------------------------------------------------------
# Fake torch fixture
# ---------------------------------------------------------------------------


class _Stream:
    """Marker object we can hand around to prove the plumbing is untouched."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_Stream({self.backend!r})"


class _Event:
    """Marker event that records which stream (if any) ``.record`` was called on."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.recorded_on: List[Any] = []

    def record(self, stream: Any = None) -> None:
        # Mirror torch.{cuda,npu}.Event.record: single optional stream arg.
        self.recorded_on.append(stream)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_Event({self.backend!r}, recorded_on={self.recorded_on!r})"


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch, log: List[Tuple[str, Any]]
) -> types.ModuleType:
    fake_torch = types.ModuleType("torch")

    fake_cuda = types.ModuleType("torch.cuda")
    _cuda_current = _Stream("cuda-current")

    def _cuda_Stream() -> _Stream:
        s = _Stream("cuda")
        log.append(("cuda.Stream", s))
        return s

    def _cuda_set_stream(s: Any) -> None:
        log.append(("cuda.set_stream", s))

    def _cuda_current_stream() -> _Stream:
        log.append(("cuda.current_stream", _cuda_current))
        return _cuda_current

    def _cuda_synchronize() -> None:
        log.append(("cuda.synchronize", None))

    def _cuda_Event() -> _Event:
        e = _Event("cuda")
        log.append(("cuda.Event", e))
        return e

    fake_cuda.Stream = _cuda_Stream  # type: ignore[attr-defined]
    fake_cuda.set_stream = _cuda_set_stream  # type: ignore[attr-defined]
    fake_cuda.current_stream = _cuda_current_stream  # type: ignore[attr-defined]
    fake_cuda.synchronize = _cuda_synchronize  # type: ignore[attr-defined]
    fake_cuda.Event = _cuda_Event  # type: ignore[attr-defined]

    fake_npu = types.ModuleType("torch.npu")
    _npu_current = _Stream("npu-current")

    def _npu_Stream() -> _Stream:
        s = _Stream("npu")
        log.append(("npu.Stream", s))
        return s

    def _npu_set_stream(s: Any) -> None:
        log.append(("npu.set_stream", s))

    def _npu_current_stream() -> _Stream:
        log.append(("npu.current_stream", _npu_current))
        return _npu_current

    def _npu_synchronize() -> None:
        log.append(("npu.synchronize", None))

    def _npu_Event() -> _Event:
        e = _Event("npu")
        log.append(("npu.Event", e))
        return e

    fake_npu.Stream = _npu_Stream  # type: ignore[attr-defined]
    fake_npu.set_stream = _npu_set_stream  # type: ignore[attr-defined]
    fake_npu.current_stream = _npu_current_stream  # type: ignore[attr-defined]
    fake_npu.synchronize = _npu_synchronize  # type: ignore[attr-defined]
    fake_npu.Event = _npu_Event  # type: ignore[attr-defined]

    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    fake_torch.npu = fake_npu  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "torch.npu", fake_npu)
    return fake_torch


def _install_torch_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))


def _install_torch_npu_import_hook(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)


# ---------------------------------------------------------------------------
# CUDA branch — all four entrypoints
# ---------------------------------------------------------------------------


def test_cuda_create_stream_calls_torch_cuda_Stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    s = dr.create_stream("cuda")

    assert isinstance(s, _Stream) and s.backend == "cuda"
    assert [name for name, _ in log] == ["cuda.Stream"]
    # Must not have imported torch_npu on the CUDA branch.
    assert "torch_npu" not in sys.modules


def test_cuda_set_stream_calls_torch_cuda_set_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    payload = _Stream("payload")
    result = dr.set_stream("cuda", payload)

    assert result is None
    assert log == [("cuda.set_stream", payload)]


def test_cuda_current_stream_calls_torch_cuda_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    got = dr.current_stream("cuda")

    assert isinstance(got, _Stream) and got.backend == "cuda-current"
    assert [name for name, _ in log] == ["cuda.current_stream"]


def test_cuda_synchronize_calls_torch_cuda_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    result = dr.synchronize_device("cuda")

    assert result is None
    assert log == [("cuda.synchronize", None)]


# ---------------------------------------------------------------------------
# NPU branch — all four entrypoints (dynamic torch_npu import required)
# ---------------------------------------------------------------------------


def test_npu_create_stream_imports_torch_npu_and_calls_torch_npu_Stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    s = dr.create_stream("npu")

    assert isinstance(s, _Stream) and s.backend == "npu"
    assert [name for name, _ in log] == ["npu.Stream"]
    assert "torch_npu" in sys.modules


def test_npu_set_stream_calls_torch_npu_set_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    payload = _Stream("payload")
    result = dr.set_stream("npu", payload)

    assert result is None
    assert log == [("npu.set_stream", payload)]


def test_npu_current_stream_calls_torch_npu_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    got = dr.current_stream("npu")

    assert isinstance(got, _Stream) and got.backend == "npu-current"
    assert [name for name, _ in log] == ["npu.current_stream"]


def test_npu_synchronize_calls_torch_npu_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    result = dr.synchronize_device("npu")

    assert result is None
    assert log == [("npu.synchronize", None)]


# ---------------------------------------------------------------------------
# CPU branch — Stream entrypoints are no-ops / None (Event covered separately)
# ---------------------------------------------------------------------------


def test_cpu_create_stream_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.create_stream("cpu") is None
    assert log == []


def test_cpu_set_stream_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.set_stream("cpu", None) is None
    # Non-None payloads are also accepted silently.
    assert dr.set_stream("cpu", _Stream("garbage")) is None
    assert log == []


def test_cpu_current_stream_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.current_stream("cpu") is None
    assert log == []


def test_cpu_synchronize_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.synchronize_device("cpu") is None
    assert log == []


def test_cpu_branch_never_touches_torch_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU calls must not trigger a torch_npu import even if the module isn't installed."""
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu_import_hook(
        monkeypatch, ImportError("torch_npu deliberately absent")
    )

    # None of these may raise.
    assert dr.create_stream("cpu") is None
    dr.set_stream("cpu", None)
    assert dr.current_stream("cpu") is None
    dr.synchronize_device("cpu")
    assert dr.create_event("cpu") is None
    dr.record_event("cpu", None, None)


# ---------------------------------------------------------------------------
# Event: CUDA branch
# ---------------------------------------------------------------------------


def test_cuda_create_event_calls_torch_cuda_Event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    ev = dr.create_event("cuda")

    assert isinstance(ev, _Event) and ev.backend == "cuda"
    assert [name for name, _ in log] == ["cuda.Event"]
    # CUDA branch must not have imported torch_npu.
    assert "torch_npu" not in sys.modules


def test_cuda_record_event_forwards_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    ev = dr.create_event("cuda")
    stream = _Stream("cuda-payload")

    result = dr.record_event("cuda", ev, stream)

    assert result is None
    # The stream we handed in was the only argument event.record saw.
    assert ev.recorded_on == [stream]


def test_cuda_record_event_defaults_to_none_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_event`` without a stream forwards ``None`` (== current stream)."""
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    ev = dr.create_event("cuda")
    dr.record_event("cuda", ev)

    assert ev.recorded_on == [None]


# ---------------------------------------------------------------------------
# Event: NPU branch (dynamic torch_npu import required)
# ---------------------------------------------------------------------------


def test_npu_create_event_imports_torch_npu_and_calls_torch_npu_Event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    ev = dr.create_event("npu")

    assert isinstance(ev, _Event) and ev.backend == "npu"
    assert [name for name, _ in log] == ["npu.Event"]
    assert "torch_npu" in sys.modules


def test_npu_record_event_forwards_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    ev = dr.create_event("npu")
    stream = _Stream("npu-payload")

    result = dr.record_event("npu", ev, stream)

    assert result is None
    assert ev.recorded_on == [stream]


def test_npu_record_event_defaults_to_none_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    ev = dr.create_event("npu")
    dr.record_event("npu", ev)

    assert ev.recorded_on == [None]


# ---------------------------------------------------------------------------
# Event: CPU branch
# ---------------------------------------------------------------------------


def test_cpu_create_event_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.create_event("cpu") is None
    assert log == []


def test_cpu_record_event_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    # None/None matches the "no event, no stream" case documented for CPU.
    assert dr.record_event("cpu", None, None) is None
    # A stray non-None payload must not raise either — record is a pure no-op.
    assert dr.record_event("cpu", _Event("garbage"), _Stream("garbage")) is None
    assert log == []


# ---------------------------------------------------------------------------
# NPU import failure — every NPU entrypoint must surface a clean RuntimeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name, args",
    [
        ("create_stream", ()),
        ("set_stream", (None,)),
        ("current_stream", ()),
        ("synchronize_device", ()),
        ("create_event", ()),
        ("record_event", (None, None)),
    ],
)
def test_npu_import_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, fn_name: str, args: tuple
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu_import_hook(monkeypatch, ImportError("no torch_npu wheel"))

    fn = getattr(dr, fn_name)
    with pytest.raises(RuntimeError, match="torch_npu"):
        fn("npu", *args)

    # Nothing should have executed on the fake torch.npu namespace.
    assert log == []


def test_npu_import_failure_preserves_non_import_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ImportError from ``import torch_npu`` also becomes a RuntimeError."""
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu_import_hook(monkeypatch, RuntimeError("ffi bind failure"))

    with pytest.raises(RuntimeError, match="torch_npu"):
        dr.create_stream("npu")


# ---------------------------------------------------------------------------
# Invalid device type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name, args",
    [
        ("create_stream", ()),
        ("set_stream", (None,)),
        ("current_stream", ()),
        ("synchronize_device", ()),
        ("create_event", ()),
        ("record_event", (None, None)),
    ],
)
def test_invalid_device_type_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, fn_name: str, args: tuple
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    fn = getattr(dr, fn_name)
    with pytest.raises(ValueError) as excinfo:
        fn("tpu", *args)  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "tpu" in msg
    # Error message must be actionable — list the supported values.
    assert "cuda" in msg and "npu" in msg and "cpu" in msg
    assert log == []


# ---------------------------------------------------------------------------
# Module-scope hygiene
# ---------------------------------------------------------------------------


def test_module_top_level_does_not_import_torch_npu() -> None:
    """``device_runtime`` must be importable on hosts without ``torch_npu``.

    ``dr`` was loaded above at module-import time; if a top-level
    ``import torch_npu`` existed we would see it in ``sys.modules`` before
    any test ran. We also check the module object itself has no ``torch_npu``
    attribute — a stronger guarantee that survives shared ``sys.modules``
    state from parametrised tests earlier in the suite.
    """
    assert not hasattr(dr, "torch_npu")


def test_public_surface_is_the_documented_functions() -> None:
    assert set(dr.__all__) == {
        "create_stream",
        "set_stream",
        "current_stream",
        "synchronize_device",
        "create_event",
        "record_event",
    }
    for name in dr.__all__:
        assert callable(getattr(dr, name))
