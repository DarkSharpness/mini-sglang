from __future__ import annotations

from typing import Any

import torch

from .utils import load_kernel_module


def _load_topk_module() -> Any:
    """
    Load the index manipulation module.
    """
    return load_kernel_module("topk.cu", "topk_kernel")


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if indices is None:
        indices = lengths.new_empty(lengths.size(0), 2048)
    _load_topk_module().fast_topk(score, indices, lengths)
    return indices


def fast_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    _load_topk_module().fast_topk_transform(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q
    )
    return dst_page_table
