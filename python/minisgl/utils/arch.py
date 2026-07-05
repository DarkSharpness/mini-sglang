from __future__ import annotations

import functools
from typing import Tuple

from .device import is_cuda_available


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """Return the current CUDA compute capability, or ``None`` on non-CUDA hosts.

    Device presence is decided by :func:`minisgl.utils.device.is_cuda_available`
    so this module contains no direct device-probe logic. On NPU/CPU hosts the
    function returns ``None`` without importing torch's CUDA APIs.
    """
    if not is_cuda_available():
        return None
    import torch
    import torch.version

    if not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
