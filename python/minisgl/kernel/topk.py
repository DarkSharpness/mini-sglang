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
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return _load_topk_module().fast_topk(score, indices, lengths)


def fast_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return _load_topk_module().fast_topk_transform(
        score, lengths, dst_page_table, src_page_table, cu_seqlens
    )
