from __future__ import annotations

import functools
from typing import Literal

DeviceType = Literal["cuda", "npu", "cpu"]

__all__ = [
    "DeviceType",
    "get_device_type",
    "is_cuda_available",
    "is_npu_available",
]


def _probe_cuda() -> bool:
    """Return True if a CUDA-enabled ``torch`` build reports at least one device.

    All failure modes (torch missing, driver missing, transient runtime errors)
    are treated as "CUDA not available" — never raised to callers.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _probe_npu() -> bool:
    """Return True if ``torch_npu`` is importable and reports at least one NPU.

    ``torch_npu`` is imported lazily via a local ``import`` statement so this
    module stays importable on hosts (e.g. macOS) where the package is absent.
    Any import- or runtime-error path degrades to False rather than propagating.
    """
    try:
        import torch_npu  # noqa: F401  (import-for-side-effect probe)
    except Exception:
        return False
    try:
        npu_mod = getattr(torch_npu, "npu", None)
        if npu_mod is None:
            return False
        is_available = getattr(npu_mod, "is_available", None)
        if is_available is None:
            return False
        return bool(is_available())
    except Exception:
        return False


@functools.cache
def is_cuda_available() -> bool:
    """Cached ``True`` iff a usable CUDA device is present. Never raises."""
    return _probe_cuda()


@functools.cache
def is_npu_available() -> bool:
    """Cached ``True`` iff a usable Ascend NPU is present. Never raises.

    Does not import ``torch_npu`` unconditionally, and does not touch device
    state — safe to call at import time on any host.
    """
    return _probe_npu()


@functools.cache
def get_device_type() -> DeviceType:
    """Return the preferred accelerator, preferring CUDA over NPU over CPU.

    Contract: no side effects, never raises, never sets or selects a device.
    Cached — call ``get_device_type.cache_clear()`` from tests when swapping
    the underlying probes via monkeypatch.
    """
    if is_cuda_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    return "cpu"
