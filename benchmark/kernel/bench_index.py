from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel import indexing
from minisgl.utils import call_if_main


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


@call_if_main(__name__)
def bench_indexing():
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # test the perf
        MEM = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),
            baseline=lambda: ref_indexing(weights, indices),
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
        )


@call_if_main(__name__)
def bench_indexing_with_mask():
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    TP = 4
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # start, length

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # test the perf
        MEM = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices, vocab_range=MASK_RANGE),
            baseline=lambda: ref_indexing(weights, indices, vocab_range=MASK_RANGE),
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
