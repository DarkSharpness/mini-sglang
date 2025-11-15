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
        cuda_wrappers=[
            ("launch_one", f"HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"HiCacheKernel<{args}>::run_all"),
        ],
    )


@lru_cache(maxsize=None)
def _jit_hicache_tma_module(
    element_size: int,
    block_quota: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, block_quota, *config)
    return load_jit(
        "hicache_tma",
        *args,
        cuda_files=["hicache_tma.cu"],
        cuda_wrappers=[("launch", f"HiCacheTMAKernel<{args}>::run")],
    )


def transfer_hicache_one_layer(
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
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    *,
    element_size: int | None = None,
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    module = _jit_hicache_module(element_size, block_quota=block_quota)
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_tma(
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
    module = _jit_hicache_tma_module(element_size, block_quota=block_quota)
    module.launch(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )
