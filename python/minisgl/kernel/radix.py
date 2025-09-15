from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .utils import load_kernel_module

if TYPE_CHECKING:
    import torch


def _load_radix_module() -> Any:
    return load_kernel_module("radix.cpp", "radix_tree")


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
