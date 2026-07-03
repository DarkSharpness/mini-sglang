"""Unit tests for :mod:`minisgl.utils.device`.

The device-probe layer must be safe to import on hosts without CUDA or
``torch_npu`` (e.g. developer macOS boxes). These tests use ``monkeypatch``
plus fake modules injected into ``sys.modules`` — no real hardware, no real
``torch_npu`` install, no reliance on the wider ``minisgl`` runtime deps.

To keep the tests hermetic on macOS the ``device`` module is loaded directly
from its file location, bypassing ``minisgl.utils.__init__`` which pulls in
``torch``, ``transformers`` and other heavyweight optional dependencies that
are not part of the device-probe contract under test.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEVICE_PATH = _REPO_ROOT / "python" / "minisgl" / "utils" / "device.py"
_ARCH_PATH = _REPO_ROOT / "python" / "minisgl" / "utils" / "arch.py"


def _load_isolated(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot build spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import device.py in isolation and expose it under its canonical dotted name
# so that ``from .device import ...`` inside arch.py resolves to the same
# object we monkeypatch below.
device = _load_isolated("minisgl.utils.device", _DEVICE_PATH)


@pytest.fixture(autouse=True)
def _reset_device_cache():
    """Clear the ``@functools.cache`` on each probe before and after every test."""
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()
    yield
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()


def _make_fake_torch_npu(available: bool) -> types.ModuleType:
    fake_npu = types.ModuleType("torch_npu.npu")
    fake_npu.is_available = lambda: available
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu = fake_npu
    return fake_torch_npu


def _install_import_hook(
    monkeypatch: pytest.MonkeyPatch, name: str, exc: BaseException
) -> None:
    """Make ``import <name>`` raise ``exc`` even if the real module exists."""
    monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = __import__

    def fake_import(module_name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if module_name == name or module_name.startswith(name + "."):
            raise exc
        return real_import(module_name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)


def test_module_imports_without_torch_npu() -> None:
    """The module must load on a host that has neither CUDA nor ``torch_npu``."""
    assert "torch_npu" not in sys.modules or sys.modules["torch_npu"] is not None
    # Public API surface exists and is callable without side effects.
    assert callable(device.get_device_type)
    assert callable(device.is_cuda_available)
    assert callable(device.is_npu_available)


def test_cuda_available_returns_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CUDA reports available, the type is 'cuda' regardless of NPU state."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: True)
    # NPU status must not matter when CUDA is present.
    monkeypatch.setattr(device, "_probe_npu", lambda: False)

    assert device.is_cuda_available() is True
    assert device.get_device_type() == "cuda"


def test_cuda_absent_and_no_torch_npu_returns_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CUDA + ``torch_npu`` missing must fall through to 'cpu' cleanly."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    _install_import_hook(monkeypatch, "torch_npu", ImportError("no torch_npu on this host"))

    assert device.is_cuda_available() is False
    assert device.is_npu_available() is False
    assert device.get_device_type() == "cpu"


def test_cuda_absent_but_npu_available_returns_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CUDA + fake ``torch_npu`` reporting a device yields 'npu'."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    fake = _make_fake_torch_npu(available=True)
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    monkeypatch.setitem(sys.modules, "torch_npu.npu", fake.npu)

    assert device.is_cuda_available() is False
    assert device.is_npu_available() is True
    assert device.get_device_type() == "npu"


def test_torch_npu_import_raises_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ``torch_npu`` that raises on import must not crash callers."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    _install_import_hook(monkeypatch, "torch_npu", RuntimeError("torch_npu ffi failure"))

    # Neither call may raise.
    assert device.is_npu_available() is False
    assert device.get_device_type() == "cpu"


def test_torch_npu_present_but_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``torch_npu`` that imports but reports ``is_available() is False`` yields cpu."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    fake = _make_fake_torch_npu(available=False)
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    monkeypatch.setitem(sys.modules, "torch_npu.npu", fake.npu)

    assert device.is_npu_available() is False
    assert device.get_device_type() == "cpu"


def test_get_device_type_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_device_type`` memoises — repeated probe swaps do not leak through."""
    monkeypatch.setattr(device, "_probe_cuda", lambda: True)
    assert device.get_device_type() == "cuda"

    # Change the underlying probe: cached value should stick until cleared.
    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    assert device.get_device_type() == "cuda"

    device.get_device_type.cache_clear()
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    _install_import_hook(monkeypatch, "torch_npu", ImportError("absent"))
    assert device.get_device_type() == "cpu"


def test_arch_module_uses_device_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """``arch.is_arch_supported`` must delegate CUDA presence to the device layer."""
    # Load arch.py in isolation; its ``from .device import is_cuda_available``
    # will resolve to the already-registered ``minisgl.utils.device`` above.
    # Also register a minimal ``minisgl.utils`` package so the relative import
    # can bind ``.device`` without executing the real ``__init__.py``.
    if "minisgl" not in sys.modules:
        pkg_minisgl = types.ModuleType("minisgl")
        pkg_minisgl.__path__ = [str(_REPO_ROOT / "python" / "minisgl")]
        sys.modules["minisgl"] = pkg_minisgl
    if "minisgl.utils" not in sys.modules:
        pkg_utils = types.ModuleType("minisgl.utils")
        pkg_utils.__path__ = [str(_REPO_ROOT / "python" / "minisgl" / "utils")]
        sys.modules["minisgl.utils"] = pkg_utils

    arch = _load_isolated("minisgl.utils.arch", _ARCH_PATH)
    arch._get_torch_cuda_version.cache_clear()

    monkeypatch.setattr(device, "_probe_cuda", lambda: False)
    _install_import_hook(monkeypatch, "torch_npu", ImportError("absent"))

    assert arch.is_arch_supported(9, 0) is False
    assert arch.is_sm90_supported() is False
    assert arch.is_sm100_supported() is False

    arch._get_torch_cuda_version.cache_clear()
