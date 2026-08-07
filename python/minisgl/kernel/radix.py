from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def warmup_radix_kernels() -> None:
    """Eagerly compile/load radix AOT kernels to avoid first-match latency."""
    _load_radix_module()


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
