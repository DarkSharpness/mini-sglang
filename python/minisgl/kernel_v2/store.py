from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KERNEL_PATH, KernelConfig

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@lru_cache(maxsize=None)
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    from tvm_ffi.cpp import load_inline

    with open(KERNEL_PATH / "jit" / "store.cu") as f:
        cuda_code = f.read()

    kernel_name = f"StoreKernel<{element_size},{config.template_args}>::run"
    cuda_code += f"\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, ({kernel_name}));"
    num_threads, max_concurrency, pdl = config
    return load_inline(
        f"index_{element_size}_{num_threads}_{max_concurrency}_{pdl}",
        cuda_sources=cuda_code,
        extra_include_paths=[str(KERNEL_PATH / "include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
