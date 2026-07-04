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
    "create_event",
    "record_event",
    "empty_device_cache",
    "reset_peak_memory_stats",
    "get_free_memory_bytes",
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


def create_event(device_type: DeviceType) -> Optional[Any]:
    """Create a fresh event on the given device type.

    * ``cuda`` → ``torch.cuda.Event()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.Event()``
    * ``cpu``  → ``None`` (CPU has no event concept)

    Gate 1.3a intentionally exposes only the default no-argument constructor.
    Timing / blocking-sync / IPC flavours are deferred to later Gates.
    """
    if device_type == "cuda":
        import torch

        return torch.cuda.Event()

    if device_type == "npu":
        _require_torch_npu()
        import torch

        return torch.npu.Event()

    if device_type == "cpu":
        return None

    raise _unsupported(device_type)


def record_event(
    device_type: DeviceType,
    event: Optional[Any],
    stream: Optional[Any] = None,
) -> None:
    """Record ``event`` on ``stream`` (or the current stream if ``None``).

    * ``cuda`` → ``event.record(stream)``
    * ``npu``  → dynamic ``import torch_npu``; then ``event.record(stream)``
    * ``cpu``  → no-op (``event`` and ``stream`` are ignored)

    ``event`` and ``stream`` are the objects previously returned by
    :func:`create_event` / :func:`create_stream` / :func:`current_stream` on
    the *same* device type. Cross-device usage is not supported and is not
    validated here — that's PyTorch's job.
    """
    if device_type == "cuda":
        # The event object itself carries the backend binding; we only need to
        # dispatch on device_type to know whether the torch_npu monkey-patch
        # must be installed before .record() is safe to call.
        import torch  # noqa: F401  (kept for symmetry with the other branches)

        event.record(stream)
        return

    if device_type == "npu":
        _require_torch_npu()
        import torch  # noqa: F401

        event.record(stream)
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def empty_device_cache(device_type: DeviceType) -> None:
    """Release cached device memory back to the allocator's free pool.

    * ``cuda`` → ``torch.cuda.empty_cache()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.empty_cache()``
    * ``cpu``  → no-op

    Gate 1.4a intentionally exposes only the plain no-argument variant used by
    :meth:`Engine._sync_get_memory`. Memory profiling / GC / device-scoped
    allocator toggles are deferred to later Gates.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.empty_cache()
        return

    if device_type == "npu":
        _require_torch_npu()
        import torch

        torch.npu.empty_cache()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def reset_peak_memory_stats(device_type: DeviceType) -> None:
    """Reset the "peak memory" counter tracked by the device allocator.

    * ``cuda`` → ``torch.cuda.reset_peak_memory_stats()``
    * ``npu``  → dynamic ``import torch_npu``; then ``torch.npu.reset_peak_memory_stats()``
    * ``cpu``  → no-op (CPU has no peak-memory counter)

    Only the no-argument, current-device form is exposed. Per-device selection
    is deferred: today :meth:`Engine._sync_get_memory` already bound the
    process to a single device via ``bind_local_device`` before touching this
    counter, so an explicit ``device=`` override would just re-encode that
    binding.
    """
    if device_type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
        return

    if device_type == "npu":
        _require_torch_npu()
        import torch

        torch.npu.reset_peak_memory_stats()
        return

    if device_type == "cpu":
        return

    raise _unsupported(device_type)


def get_free_memory_bytes(device_type: DeviceType, device: Any) -> int:
    """Return the currently-free device memory in bytes.

    * ``cuda`` → ``torch.cuda.mem_get_info(device)[0]`` → ``int``
    * ``npu``  → dynamic ``import torch_npu``; then
      ``torch.npu.mem_get_info(device)[0]`` → ``int``
    * ``cpu``  → raises :class:`NotImplementedError` — CPU has no dedicated
      device memory pool distinct from host RAM, and Gate 1.5a deliberately
      does not shim in a ``psutil`` fallback.

    ``device`` is forwarded verbatim to the vendor ``mem_get_info`` call. It
    accepts whatever that call accepts today (``int`` index, ``torch.device``,
    ``str``) — this dispatch layer does not narrow the contract.

    Only the free half of the tuple is returned; ``total_bytes`` is not exposed
    on purpose (Engine's memory bookkeeping never needs it).
    """
    if device_type == "cuda":
        import torch

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)

    if device_type == "npu":
        _require_torch_npu()
        import torch

        free_bytes, _total_bytes = torch.npu.mem_get_info(device)
        return int(free_bytes)

    if device_type == "cpu":
        raise NotImplementedError(
            "free device memory query is not implemented for CPU"
        )

    raise _unsupported(device_type)
