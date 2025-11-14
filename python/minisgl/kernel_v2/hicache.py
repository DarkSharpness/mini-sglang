from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=1024, max_occupancy=1, use_pdl=False)
DEFAULT_BLOCK_QUOTA = 4
INF = 1 << 30


@lru_cache(maxsize=None)
def _jit_hicache_module(
    element_size: int,
    block_quota: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, block_quota, *config)
    return load_jit(
        "hicache",
        *args,
        cuda_files=["hicache.cu"],
        cuda_wrappers=[("launch", f"HiCacheKernel<{args}>::run")],
    )


def transfer_hicache(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    element_size = k_cache_dst.element_size() * k_cache_dst.size(1)
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    module = _jit_hicache_module(element_size, block_quota=block_quota)
    module.launch(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )
