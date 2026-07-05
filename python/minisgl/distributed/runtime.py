from __future__ import annotations

import os
from dataclasses import dataclass

from ..utils.device import DeviceType, get_device_type
from .backend import DistributedBackend, get_distributed_backend

__all__ = ["DistributedRuntime", "bind_local_device", "initialize_distributed_from_env"]


@dataclass(frozen=True)
class DistributedRuntime:
    """Resolved single-process view of the distributed job.

    Frozen so callers can safely stash it on module- or engine-level state
    without worrying about mutation. Populated exclusively by
    :func:`initialize_distributed_from_env` — no other constructors are
    supported.
    """

    device_type: DeviceType
    backend: DistributedBackend
    local_rank: int
    rank: int
    world_size: int
    device: str


_REQUIRED_ENV = ("LOCAL_RANK", "RANK", "WORLD_SIZE")


def _read_int_env(name: str) -> int:
    """Parse a required int environment variable; raise ``RuntimeError`` otherwise."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        raise RuntimeError(
            f"required environment variable {name!r} is not set; "
            f"expected all of {list(_REQUIRED_ENV)} to be set by the launcher"
        )
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f"environment variable {name!r} must be an integer, got {raw!r}"
        ) from None


def bind_local_device(device_type: DeviceType, local_rank: int) -> str:
    """Bind the current process to its local device and return the device string.

    This is the shared, side-effect-scoped device-binding primitive used by
    both :func:`initialize_distributed_from_env` and the engine bootstrap.
    It does **not** touch ``torch.distributed`` — callers remain in charge of
    process-group init.

    * ``cuda``: ``torch.cuda.set_device(local_rank)`` → ``"cuda:{local_rank}"``
    * ``npu``: dynamic ``import torch_npu`` (which patches ``torch.npu`` onto
      ``torch``), then ``torch.npu.set_device(local_rank)`` → ``"npu:{local_rank}"``
    * ``cpu``: no-op → ``"cpu"``

    ``torch_npu`` is *never* imported at module scope; only this branch, only
    when the resolved device is ``npu``. Callers on a macOS / CPU-only host
    may import this module freely.

    Raises:
        RuntimeError: if ``device_type == "npu"`` but ``torch_npu`` cannot be
            imported.
        ValueError: if ``device_type`` is not one of ``"cuda"``, ``"npu"``,
            ``"cpu"``.
    """
    if device_type == "cuda":
        import torch  # lazy: only touched on CUDA hosts

        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"

    if device_type == "npu":
        # Safe dynamic import — kept out of module scope so macOS / CPU-only
        # hosts stay importable.
        try:
            import torch_npu  # noqa: F401  (import-for-side-effect: patches torch.npu)
        except Exception as exc:
            raise RuntimeError(
                "device_type is 'npu' but 'torch_npu' could not be imported; "
                "install the matching torch_npu wheel on the Ascend host"
            ) from exc

        import torch

        torch.npu.set_device(local_rank)
        return f"npu:{local_rank}"

    if device_type == "cpu":
        return "cpu"

    raise ValueError(f"unsupported device_type for binding: {device_type!r}")


def initialize_distributed_from_env() -> DistributedRuntime:
    """Read env vars, bind the local device, and initialise ``torch.distributed``.

    Steps, in order:

    1. Parse ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` from the environment
       and validate their ranges. Env-var errors surface as ``RuntimeError``
       (missing/non-integer) or ``ValueError`` (out of range).
    2. Resolve ``device_type`` via :func:`get_device_type` and ``backend``
       via :func:`get_distributed_backend`.
    3. Guard against double init: raise ``RuntimeError`` if
       ``torch.distributed.is_initialized()`` is already true. The device is
       not touched in that case.
    4. Bind the local device (``torch.cuda.set_device`` / ``torch.npu.set_device``
       / no-op).
    5. Call ``torch.distributed.init_process_group(backend=backend)`` with
       no ``init_method``, no ``rank``/``world_size`` overrides — the launcher
       (torchrun, MPI, etc.) is expected to supply them via env vars.
    6. Return a frozen :class:`DistributedRuntime`.

    This function does not run any collective, does not call ``barrier``,
    does not register any destroy hook, and does not touch PyNCCL or the
    engine layer. It is idempotent-guarded but not idempotent itself — call
    it exactly once per process.
    """
    local_rank = _read_int_env("LOCAL_RANK")
    rank = _read_int_env("RANK")
    world_size = _read_int_env("WORLD_SIZE")

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK must satisfy 0 <= RANK < WORLD_SIZE (WORLD_SIZE={world_size}), "
            f"got RANK={rank}"
        )
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")

    device_type: DeviceType = get_device_type()
    backend: DistributedBackend = get_distributed_backend(device_type)

    import torch.distributed as dist  # lazy: import only at initialisation time

    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialised; "
            "initialize_distributed_from_env() must run at most once per process"
        )

    device = bind_local_device(device_type, local_rank)
    dist.init_process_group(backend=backend)

    return DistributedRuntime(
        device_type=device_type,
        backend=backend,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        device=device,
    )
