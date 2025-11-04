from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple

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
        f"minisgl__store_{element_size}_{num_threads}_{max_concurrency}_{pdl}",
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


@lru_cache(maxsize=1)
def _lru_get_pos(pos_list: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    import torch

    """
    This position can be hopefully reused, especially for patterns that
    a load often happens right after a store.
    """
    pos_tensor = torch.tensor(pos_list, device="cpu", dtype=torch.int32, pin_memory=True)
    return pos_tensor.to(device, non_blocking=True)


def store_decode_indices(
    page_table: torch.Tensor,
    indices: torch.Tensor,
    pos: List[Tuple[int, int]],
) -> None:
    """
    Write decode indices to the given page_table.
    """
    page_table_len = page_table.size(1)
    flattened_table = page_table.view(-1)
    pos_list = tuple(p[0] * page_table_len + p[1] for p in pos)
    pos_tensor = _lru_get_pos(tuple(pos_list), page_table.device)
    flattened_table[pos_tensor] = indices


def load_decode_indices(
    page_table: torch.Tensor,
    pos: List[Tuple[int, int]],
) -> torch.Tensor:
    page_table_len = page_table.size(1)
    flattened_table = page_table.view(-1)
    pos_list = tuple(p[0] * page_table_len + p[1] for p in pos)
    pos_tensor = _lru_get_pos(tuple(pos_list), page_table.device)
    return flattened_table[pos_tensor]
