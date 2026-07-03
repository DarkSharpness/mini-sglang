"""Device-runtime stream primitives shared across CUDA and NPU.

This module is a **pure dispatch layer** over ``torch.cuda`` / ``torch.npu``
for the four stream entrypoints the engine currently touches — creation,
binding as the current stream, reading the current stream, and device-wide
synchronization. Every call routes through here so that when Gate 1.2+ ports
the engine off the raw ``torch.cuda.*`` API only this file needs to grow new
branches.

Deliberately NOT in scope for Gate 1.2a:

* ``Event`` — has its own Gate.
* Stream context managers (``with stream:``) — has its own Gate.
* Wiring these primitives into ``Engine`` — has its own Gate.
* Memory / graph / kernel abstractions — separate Gates.

Design notes:

* No top-level ``import torch_npu``. The NPU branch imports it lazily inside
  each function so macOS / CPU-only hosts stay importable.
* CPU is a real, fully-supported device type: ``create_stream("cpu")`` returns
  ``None`` and the other three CPU calls are no-ops. This matches how
  :mod:`torch` itself treats CPU — there is no stream object.
* Unknown ``device_type`` values surface as ``ValueError`` — the same error
  shape :func:`minisgl.distributed.runtime.bind_local_device` and
  :func:`minisgl.distributed.backend.get_distributed_backend` use.
"""
from __future__ import annotations

from typing import Any, Optional

from .device import DeviceType

__all__ = [
    "create_stream",
    "set_stream",
    "current_stream",
    "synchronize_device",
]


def _require_torch_npu() -> None:
    """Dynamic ``import torch_npu`` with a clean error message on failure.

    ``torch_npu`` monkey-patches ``torch.npu`` at import time; without it the
    ``torch.npu`` namespace is unusable even on a real Ascend host. We centralise
    the import here so every NPU branch gets the same actionable RuntimeError.
    """
    try:
        import torch_npu  # noqa: F401  (import-for-side-effect: patches torch.npu)
    except Exception as exc:
        raise RuntimeError(
            "device_type is 'npu' but 'torch_npu' could not be imported; "
            "install the matching torch_npu wheel on the Ascend host"
        ) from exc


def _unsupported(device_type: Any) -> ValueError:
    return ValueError(
        f"unsupported device_type for device_runtime: {device_type!r}; "
        f"expected one of: cpu, cuda, npu"
    )


def create_stream(device_type: DeviceType) -> Optional[Any]:
    """Create a fresh stream on the given device type.

    * ``cuda`` → ``torch.cuda.Stream()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.Stream()``
    * ``cpu``  → ``None`` (CPU has no stream concept)
    """
    if device_type == "cuda":
        import torch  # lazy: only touched on CUDA hosts

        return torch.cuda.Stream()

    if device_type == "npu":
        _require_torch_npu()
        import torch

        return torch.npu.Stream()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def set_stream(device_type: DeviceType, stream: Optional[Any]) -> None:
    """Bind ``stream`` as the current stream on the given device type.

    * ``cuda`` → ``torch.cuda.set_stream(stream)``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.set_stream(stream)``
    * ``cpu``  → no-op regardless of ``stream``

    CPU accepts ``None`` (or any value) silently — matches ``create_stream``'s
    return contract.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.set_stream(stream)
        return

    if device_type == "npu":
        _require_torch_npu()
        import torch

        torch.npu.set_stream(stream)
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def current_stream(device_type: DeviceType) -> Optional[Any]:
    """Return the current stream on the given device type.

    * ``cuda`` → ``torch.cuda.current_stream()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.current_stream()``
    * ``cpu``  → ``None``
    """
    if device_type == "cuda":
        import torch

        return torch.cuda.current_stream()

    if device_type == "npu":
        _require_torch_npu()
        import torch

        return torch.npu.current_stream()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def synchronize_device(device_type: DeviceType) -> None:
    """Block until all previously-queued work on the given device completes.

    * ``cuda`` → ``torch.cuda.synchronize()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.synchronize()``
    * ``cpu``  → no-op
    """
    if device_type == "cuda":
        import torch

        torch.cuda.synchronize()
        return

    if device_type == "npu":
        _require_torch_npu()
        import torch

        torch.npu.synchronize()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)
