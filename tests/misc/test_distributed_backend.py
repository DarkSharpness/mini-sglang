"""Unit tests for :mod:`minisgl.distributed.backend`.

The backend selector must be a pure lookup — no torch, no ``torch_npu``, no
process-group init, no environment reads. These tests exercise the mapping
directly and use ``monkeypatch`` to swap the underlying device probes.

Like :mod:`tests.misc.test_device`, the modules under test are loaded via
``importlib.util`` so the suite runs on macOS without the heavyweight optional
runtime dependencies (``torch``, ``huggingface_hub``, ...) referenced by the
package ``__init__`` files.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY_ROOT = _REPO_ROOT / "python"
_DEVICE_PATH = _PY_ROOT / "minisgl" / "utils" / "device.py"
_BACKEND_PATH = _PY_ROOT / "minisgl" / "distributed" / "backend.py"


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


# Register a minimal ``minisgl`` / ``minisgl.utils`` / ``minisgl.distributed``
# package skeleton so the relative import ``from ..utils.device import ...``
# inside backend.py resolves without executing the real ``__init__`` files
# (which pull in torch, huggingface_hub, and the PyNCCL wrapper).
_install_package_stub("minisgl", _PY_ROOT / "minisgl")
_install_package_stub("minisgl.utils", _PY_ROOT / "minisgl" / "utils")
_install_package_stub("minisgl.distributed", _PY_ROOT / "minisgl" / "distributed")

device = _load_isolated("minisgl.utils.device", _DEVICE_PATH)
backend = _load_isolated("minisgl.distributed.backend", _BACKEND_PATH)


@pytest.fixture(autouse=True)
def _reset_device_cache():
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()
    yield
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()


def test_explicit_cuda_maps_to_nccl() -> None:
    assert backend.get_distributed_backend("cuda") == "nccl"


def test_explicit_npu_maps_to_hccl() -> None:
    assert backend.get_distributed_backend("npu") == "hccl"


def test_explicit_cpu_maps_to_gloo() -> None:
    assert backend.get_distributed_backend("cpu") == "gloo"


def test_none_defers_to_device_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_probe_cuda", lambda: True)
    monkeypatch.setattr(device, "_probe_npu", lambda: False)

    assert backend.get_distributed_backend() == "nccl"
    assert backend.get_distributed_backend(None) == "nccl"


def test_none_defers_to_device_probe_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    monkeypatch.setattr(device, "_probe_npu", lambda: True)

    assert backend.get_distributed_backend() == "hccl"


def test_none_defers_to_device_probe_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    monkeypatch.setattr(device, "_probe_npu", lambda: False)

    assert backend.get_distributed_backend() == "gloo"


def test_explicit_value_does_not_reprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the caller supplies ``device_type`` we must NOT hit the probes."""
    calls: list[str] = []

    def spy_get_device_type() -> str:  # pragma: no cover - must not fire
        calls.append("get_device_type")
        return "cuda"

    def spy_probe_cuda() -> bool:  # pragma: no cover - must not fire
        calls.append("_probe_cuda")
        return True

    def spy_probe_npu() -> bool:  # pragma: no cover - must not fire
        calls.append("_probe_npu")
        return True

    monkeypatch.setattr(device, "get_device_type", spy_get_device_type)
    monkeypatch.setattr(device, "_probe_cuda", spy_probe_cuda)
    monkeypatch.setattr(device, "_probe_npu", spy_probe_npu)
    # backend.py imported the symbol at load time; patch its local binding too.
    monkeypatch.setattr(backend, "get_device_type", spy_get_device_type)

    assert backend.get_distributed_backend("npu") == "hccl"
    assert backend.get_distributed_backend("cuda") == "nccl"
    assert backend.get_distributed_backend("cpu") == "gloo"
    assert calls == []


def test_invalid_device_type_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        backend.get_distributed_backend("tpu")  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "tpu" in msg
    # Error message must be actionable — list the supported values.
    assert "cuda" in msg and "npu" in msg and "cpu" in msg


def test_invalid_device_type_from_probe_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a corrupt probe result surfaces as ValueError, not KeyError."""
    monkeypatch.setattr(backend, "get_device_type", lambda: "wat")

    with pytest.raises(ValueError, match="wat"):
        backend.get_distributed_backend()


def test_backend_module_does_not_import_torch_npu() -> None:
    """The lookup module must stay import-free of ``torch_npu``."""
    # backend.py has already been executed above; if it imported torch_npu at
    # module scope we would see it registered in sys.modules.
    assert "torch_npu" not in sys.modules
