from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
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


def is_sm120_supported() -> bool:
    # Consumer Blackwell (sm120) numerically passes the sm100 check but doesn't support
    # trtllm-gen/FA4 kernels, which target datacenter Blackwell (sm100/B200) only.
    return is_arch_supported(12, 0)
