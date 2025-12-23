from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

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
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


@lru_cache(maxsize=None)
def _jit_store_mla_module(
    kv_c_size: int,
    k_rope_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(kv_c_size, k_rope_size, *config)
    return load_jit(
        "store_mla",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreMLAKernel<{args}>::run")],
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


def store_mla_cache(
    kv_buffer: torch.Tensor,  # Single buffer containing both kv_c and k_rope
    indices: torch.Tensor,
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
) -> None:
    kv_buffer = kv_buffer.view(-1, kv_buffer.shape[-1])
    kv_c_size = kv_c.shape[-1] * kv_c.element_size()
    k_rope_size = k_rope.shape[-1] * k_rope.element_size()
    module = _jit_store_mla_module(kv_c_size, k_rope_size)
    module.launch(kv_buffer, indices, kv_c, k_rope)
