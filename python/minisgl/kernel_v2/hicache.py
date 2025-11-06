from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KERNEL_PATH, KernelConfig

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=1024, max_occupancy=2, use_pdl=False)
DEFAULT_BLOCK_QUOTA = 4
INF = 1 << 30


@lru_cache(maxsize=None)
def _jit_hicache_module(
    element_size: int,
    block_quota: int | None = None,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    from tvm_ffi.cpp import load_inline

    with open(KERNEL_PATH / "jit" / "hicache.cu") as f:
        cuda_code = f.read()

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    kernel_name = f"HicacheKernel<{element_size},{block_quota},{config.template_args}>::run"
    cuda_code += f"\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, ({kernel_name}));"
    num_threads, max_concurrency, pdl = config
    return load_inline(
        f"minisgl__hicache_{element_size}_{block_quota}_{num_threads}_{max_concurrency}_{pdl}",
        cuda_sources=cuda_code,
        extra_include_paths=[str(KERNEL_PATH / "include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def transfer_hicache(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    block_quota: int | None = None,  # can be tuned for less interference
    split_limit: int | None = None,  # can be tuned for better performance
) -> None:
    element_size = k_cache_dst.element_size() * k_cache_dst.size(1)
    module = _jit_hicache_module(element_size, block_quota=block_quota)
    if split_limit is None:
        split_limit = INF

    module.launch(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
        split_limit,
    )
