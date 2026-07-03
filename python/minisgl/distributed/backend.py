from __future__ import annotations

from typing import Literal

from ..utils.device import DeviceType, get_device_type

DistributedBackend = Literal["nccl", "hccl", "gloo"]

__all__ = ["DistributedBackend", "get_distributed_backend"]


# Single source of truth for the device → backend string mapping. Kept as a
# plain dict so it is trivially inspectable by callers/tests, but the module's
# only public contract is ``get_distributed_backend``.
_DEVICE_TO_BACKEND: dict[str, DistributedBackend] = {
    "cuda": "nccl",
    "npu": "hccl",
    "cpu": "gloo",
}


def get_distributed_backend(device_type: DeviceType | None = None) -> DistributedBackend:
    """Return the ``torch.distributed`` backend string for the given device.

    Mapping::

        cuda -> nccl
        npu  -> hccl
        cpu  -> gloo

    When ``device_type`` is ``None`` the function defers to
    :func:`minisgl.utils.device.get_device_type` (which itself is cached and
    side-effect free). When an explicit value is supplied the device layer is
    **not** re-probed — the caller's choice is honored verbatim.

    This helper is a pure lookup: it does not import ``torch_npu``, does not
    call ``torch.distributed.init_process_group``, does not touch any device
    or environment variable, and does not read rank/world-size state.

    Raises:
        ValueError: if ``device_type`` is not one of ``"cuda"``, ``"npu"``,
            ``"cpu"``.
    """
    resolved = device_type if device_type is not None else get_device_type()
    try:
        return _DEVICE_TO_BACKEND[resolved]
    except KeyError:
        supported = ", ".join(sorted(_DEVICE_TO_BACKEND))
        raise ValueError(
            f"unsupported device_type {resolved!r}; expected one of: {supported}"
        ) from None
