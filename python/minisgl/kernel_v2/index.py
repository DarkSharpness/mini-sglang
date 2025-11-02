from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Tuple

from .utils import KERNEL_PATH, KernelConfig

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@lru_cache(maxsize=None)
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    from tvm_ffi.cpp import load_inline

    with open(KERNEL_PATH / "jit" / "index.cu") as f:
        cuda_code = f.read()

    kernel_name = f"IndexKernel<{element_size},{num_splits},{config.template_args}>::run"
    cuda_code += f"\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, ({kernel_name}));"
    num_threads, max_concurrency, pdl = config
    return load_inline(
        f"index_{element_size}_{num_splits}_{num_threads}_{max_concurrency}_{pdl}",
        cuda_sources=cuda_code,
        extra_include_paths=[str(KERNEL_PATH / "include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])
    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
