from __future__ import annotations

from typing import Any, Tuple

import torch

from .utils import load_kernel_module


def _load_index_module() -> Any:
    """
    Load the index manipulation module.
    """
    return load_kernel_module("index.cu", "indexing_kernel")


def fused_indexing(
    input: torch.Tensor,
    index: torch.Tensor,
    vocab_range: Tuple[int, int],
    output: torch.Tensor | None = None,
    block_size: int = 256,
    strict: bool = False,
) -> torch.Tensor:
    """
    Perform fused indexing operation on the input tensor using the provided indices.
    The output tensor is modified in place.
    """
    module = _load_index_module()
    if output is None:
        import torch

        output = torch.empty(
            (index.size(0), input.size(1)),
            dtype=input.dtype,
            device=input.device,
        )

    module.fused_indexing(
        output,
        input,
        index,
        vocab_range,
        block_size,
        strict,
    )
    return output
