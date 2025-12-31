from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F
import pytest

from minisgl.kernel import indexing


def ref_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if vocab_range is not None:
        start, length = vocab_range
        assert length <= weights.shape[0]
        indices = indices - start
        indices_mask = (indices < 0) | (indices >= length)
        indices[indices_mask] = 0  # set out-of-vocab indices to zero
        result = F.embedding(indices, weights)
        result[indices_mask] = 0
        return result
    else:
        return F.embedding(indices, weights)


@pytest.mark.cuda
def test_indexing_correctness(cuda_device):
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device=cuda_device, dtype=torch.float16)

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device=cuda_device, dtype=torch.int32)

        # first test the correctness
        result = indexing(
            weights,
            indices,
        )
        expected = ref_indexing(
            weights,
            indices,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"


@pytest.mark.cuda
def test_indexing_with_mask(cuda_device):
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    TP = 4
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device=cuda_device, dtype=torch.float16)

    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # start, length

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device=cuda_device, dtype=torch.int32)

        # first test the correctness
        result = indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        expected = ref_indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"
