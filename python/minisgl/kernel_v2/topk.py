from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KERNEL_PATH

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_topk_module() -> Module:
    from tvm_ffi.cpp import load_inline

    with open(KERNEL_PATH / "src" / "topk_2048.cu") as f:
        cuda_code = f.read()

    return load_inline(
        "minisgl__topk_2048",
        cuda_sources=cuda_code,
        extra_include_paths=[str(KERNEL_PATH / "include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if indices is None:
        indices = lengths.new_empty(lengths.size(0), 2048)
    _load_topk_module().topk(score, lengths, indices)
    return indices


def fast_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    _load_topk_module().topk_transform(score, lengths, dst_page_table, src_page_table, cu_seqlens_q)
    return dst_page_table
