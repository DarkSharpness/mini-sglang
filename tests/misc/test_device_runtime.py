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

    def _cuda_empty_cache() -> None:
        log.append(("cuda.empty_cache", None))

    def _cuda_reset_peak_memory_stats() -> None:
        log.append(("cuda.reset_peak_memory_stats", None))

    def _cuda_mem_get_info(device: Any) -> Tuple[int, int]:
        # Return (free, total). We stamp both slots into the log so tests can
        # prove the device arg was forwarded and that free is the [0] slot.
        log.append(("cuda.mem_get_info", device))
        return (1234, 5678)

    class _CudaStreamCtx:
        """Fake context manager returned by ``torch.cuda.stream(stream)``."""

        def __init__(self, stream: Any) -> None:
            self.stream = stream
            log.append(("cuda.stream", stream))

        def __enter__(self) -> "_CudaStreamCtx":
            log.append(("cuda.stream.__enter__", self.stream))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            log.append(("cuda.stream.__exit__", self.stream))

    fake_cuda.Stream = _cuda_Stream  # type: ignore[attr-defined]
    fake_cuda.set_stream = _cuda_set_stream  # type: ignore[attr-defined]
    fake_cuda.current_stream = _cuda_current_stream  # type: ignore[attr-defined]
    fake_cuda.synchronize = _cuda_synchronize  # type: ignore[attr-defined]
    fake_cuda.Event = _cuda_Event  # type: ignore[attr-defined]
    fake_cuda.empty_cache = _cuda_empty_cache  # type: ignore[attr-defined]
    fake_cuda.reset_peak_memory_stats = _cuda_reset_peak_memory_stats  # type: ignore[attr-defined]
    fake_cuda.mem_get_info = _cuda_mem_get_info  # type: ignore[attr-defined]
    fake_cuda.stream = _CudaStreamCtx  # type: ignore[attr-defined]

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

    def _npu_empty_cache() -> None:
        log.append(("npu.empty_cache", None))

    def _npu_reset_peak_memory_stats() -> None:
        log.append(("npu.reset_peak_memory_stats", None))

    def _npu_mem_get_info(device: Any) -> Tuple[int, int]:
        log.append(("npu.mem_get_info", device))
        return (7777, 9999)

    class _NpuStreamCtx:
        """Fake context manager returned by ``torch.npu.stream(stream)``."""

        def __init__(self, stream: Any) -> None:
            self.stream = stream
            log.append(("npu.stream", stream))

        def __enter__(self) -> "_NpuStreamCtx":
            log.append(("npu.stream.__enter__", self.stream))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            log.append(("npu.stream.__exit__", self.stream))

    fake_npu.Stream = _npu_Stream  # type: ignore[attr-defined]
    fake_npu.set_stream = _npu_set_stream  # type: ignore[attr-defined]
    fake_npu.current_stream = _npu_current_stream  # type: ignore[attr-defined]
    fake_npu.synchronize = _npu_synchronize  # type: ignore[attr-defined]
    fake_npu.Event = _npu_Event  # type: ignore[attr-defined]
    fake_npu.empty_cache = _npu_empty_cache  # type: ignore[attr-defined]
    fake_npu.reset_peak_memory_stats = _npu_reset_peak_memory_stats  # type: ignore[attr-defined]
    fake_npu.mem_get_info = _npu_mem_get_info  # type: ignore[attr-defined]
    fake_npu.stream = _NpuStreamCtx  # type: ignore[attr-defined]

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


# ---------------------------------------------------------------------------
# stream_context — cuda / npu / cpu dispatch
# ---------------------------------------------------------------------------


def test_cuda_stream_context_calls_torch_cuda_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    payload = _Stream("cuda-payload")
    ctx = dr.stream_context("cuda", payload)

    # Construction dispatched to torch.cuda.stream(payload) — no npu import.
    assert [name for name, _ in log] == ["cuda.stream"]
    assert "torch_npu" not in sys.modules

    # The returned object is a real context manager, and re-entering it works.
    with ctx:
        pass
    with ctx:
        pass

    assert [name for name, _ in log] == [
        "cuda.stream",
        "cuda.stream.__enter__",
        "cuda.stream.__exit__",
        "cuda.stream.__enter__",
        "cuda.stream.__exit__",
    ]


def test_npu_stream_context_imports_torch_npu_and_calls_torch_npu_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    payload = _Stream("npu-payload")
    ctx = dr.stream_context("npu", payload)

    # Construction dispatched to torch.npu.stream(payload). Also, torch.cuda.stream
    # must not have been touched — NPU hosts have no CUDA.
    assert [name for name, _ in log] == ["npu.stream"]
    assert "torch_npu" in sys.modules

    with ctx:
        pass
    assert [name for name, _ in log] == [
        "npu.stream",
        "npu.stream.__enter__",
        "npu.stream.__exit__",
    ]


def test_cpu_stream_context_returns_nullcontext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    ctx = dr.stream_context("cpu", None)

    with ctx:
        pass
    with ctx:  # nullcontext is safely re-entrant
        pass
    # CPU branch must not touch torch.cuda.stream or torch.npu.stream.
    assert log == []


def test_stream_context_cpu_never_touches_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU stream_context must not trigger a torch_npu import."""
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu_import_hook(
        monkeypatch, ImportError("torch_npu deliberately absent")
    )
    # Must not raise even though torch_npu import is booby-trapped.
    with dr.stream_context("cpu", None):
        pass


def test_stream_context_rejects_unknown_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    with pytest.raises(ValueError, match="unsupported device_type"):
        dr.stream_context("mps", None)  # type: ignore[arg-type]


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
    dr.empty_device_cache("cpu")
    dr.reset_peak_memory_stats("cpu")


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
# Memory maintenance: CUDA branch
# ---------------------------------------------------------------------------


def test_cuda_empty_device_cache_calls_torch_cuda_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    result = dr.empty_device_cache("cuda")

    assert result is None
    assert log == [("cuda.empty_cache", None)]
    # CUDA branch must not have imported torch_npu.
    assert "torch_npu" not in sys.modules


def test_cuda_reset_peak_memory_stats_calls_torch_cuda_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    result = dr.reset_peak_memory_stats("cuda")

    assert result is None
    assert log == [("cuda.reset_peak_memory_stats", None)]
    assert "torch_npu" not in sys.modules


# ---------------------------------------------------------------------------
# Memory maintenance: NPU branch (dynamic torch_npu import required)
# ---------------------------------------------------------------------------


def test_npu_empty_device_cache_imports_torch_npu_and_calls_torch_npu_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    result = dr.empty_device_cache("npu")

    assert result is None
    assert log == [("npu.empty_cache", None)]
    assert "torch_npu" in sys.modules


def test_npu_reset_peak_memory_stats_imports_torch_npu_and_calls_torch_npu_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    result = dr.reset_peak_memory_stats("npu")

    assert result is None
    assert log == [("npu.reset_peak_memory_stats", None)]
    assert "torch_npu" in sys.modules


# ---------------------------------------------------------------------------
# Memory maintenance: CPU branch
# ---------------------------------------------------------------------------


def test_cpu_empty_device_cache_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.empty_device_cache("cpu") is None
    assert log == []


def test_cpu_reset_peak_memory_stats_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    assert dr.reset_peak_memory_stats("cpu") is None
    assert log == []


# ---------------------------------------------------------------------------
# Free memory: CUDA branch
# ---------------------------------------------------------------------------


def test_cuda_get_free_memory_bytes_calls_torch_cuda_mem_get_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    result = dr.get_free_memory_bytes("cuda", 3)

    # Fake returns (1234, 5678); we must expose the *free* slot as an int.
    assert result == 1234
    assert isinstance(result, int)
    assert log == [("cuda.mem_get_info", 3)]
    # CUDA branch must not have imported torch_npu.
    assert "torch_npu" not in sys.modules


def test_cuda_get_free_memory_bytes_forwards_arbitrary_device_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``device`` param is passed through verbatim — no coercion."""
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    marker = object()  # anything hashable that torch would normally accept
    dr.get_free_memory_bytes("cuda", marker)

    assert log == [("cuda.mem_get_info", marker)]


def test_cuda_get_free_memory_bytes_return_type_is_native_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The return value is coerced to a real ``int``, never a torch/np scalar.

    Reinstall the fake with a mem_get_info that returns numpy-style ints via a
    subclass; the wrapper must still hand back a plain ``int``.
    """

    class _WeirdInt(int):
        pass

    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    # Overwrite mem_get_info with a version that returns a non-plain int.
    import torch  # comes from the fake we just installed

    def _weird(device: Any):
        log.append(("cuda.mem_get_info", device))
        return (_WeirdInt(42), _WeirdInt(100))

    monkeypatch.setattr(torch.cuda, "mem_get_info", _weird)

    result = dr.get_free_memory_bytes("cuda", 0)

    assert result == 42
    assert type(result) is int  # not _WeirdInt — the wrapper coerced it


# ---------------------------------------------------------------------------
# Free memory: NPU branch (dynamic torch_npu import required)
# ---------------------------------------------------------------------------


def test_npu_get_free_memory_bytes_imports_torch_npu_and_calls_torch_npu_mem_get_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    result = dr.get_free_memory_bytes("npu", 5)

    assert result == 7777
    assert isinstance(result, int)
    assert log == [("npu.mem_get_info", 5)]
    assert "torch_npu" in sys.modules


def test_npu_get_free_memory_bytes_forwards_arbitrary_device_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    marker = object()
    dr.get_free_memory_bytes("npu", marker)

    assert log == [("npu.mem_get_info", marker)]


# ---------------------------------------------------------------------------
# Free memory: CPU branch — must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_cpu_get_free_memory_bytes_raises_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: List[Tuple[str, Any]] = []
    _install_fake_torch(monkeypatch, log)

    with pytest.raises(NotImplementedError, match="CPU"):
        dr.get_free_memory_bytes("cpu", None)

    # Nothing on the fake torch namespaces was touched.
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
        ("empty_device_cache", ()),
        ("reset_peak_memory_stats", ()),
        ("get_free_memory_bytes", (0,)),
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
        ("empty_device_cache", ()),
        ("reset_peak_memory_stats", ()),
        ("get_free_memory_bytes", (0,)),
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
        "stream_context",
        "synchronize_device",
        "create_event",
        "record_event",
        "empty_device_cache",
        "reset_peak_memory_stats",
        "get_free_memory_bytes",
    }
    for name in dr.__all__:
        assert callable(getattr(dr, name))
