from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .utils import KERNEL_PATH

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=None)
def _load_radix_module() -> Any:
    from tvm_ffi.cpp import load_inline

    with open(KERNEL_PATH / "src" / "radix.cpp") as f:
        cuda_code = f.read()

    return load_inline(
        "minisgl__radix",
        cuda_sources=cuda_code,
        extra_include_paths=[str(KERNEL_PATH / "include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
