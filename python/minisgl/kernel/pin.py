from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from .utils import load_kernel_module

if TYPE_CHECKING:
    import torch


def _load_pin_module() -> Any:
    """
    Load the pin module for memory pinning operations.
    """
    return load_kernel_module(
        "pin.cu",
        "pin_memory_allocator",
        ldflags=("-lnuma",),  # need libnuma for NUMA support
    )


def create_pin_tensor(
    shape: Iterable[int],
    dtype: torch.dtype,
    write_combine: bool = False,
    numa: int | None = None,
) -> torch.Tensor:
    """
    Create a pinned memory tensor using the pin module.
    This tensor can be used for efficient data transfer between CPU and GPU.
    """
    module = _load_pin_module()
    # multiply all dimensions to get the total number of elements
    total_elements = 1
    shape_tuple = tuple(shape)
    for dim in shape_tuple:
        total_elements *= dim
    tensor = module.make_pin_tensor(total_elements, dtype, write_combine, numa)
    assert tensor.is_pinned(), "Failed to create a pinned tensor"
    return tensor.view(shape_tuple)


def get_numa_count() -> int:
    """
    Get the number of NUMA nodes available on the system.
    """
    module = _load_pin_module()
    return module.numa_count()
