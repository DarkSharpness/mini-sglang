from __future__ import annotations

import torch
from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel import store_cache
from minisgl.utils import call_if_main


@call_if_main(__name__)
def bench_store_cache():
    HEAD_SIZE = 128
    NUM_TOKENS = 1048576  # 1M
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    for bs in [2**n for n in range(0, 16)]:
        # NOTE: we cannot tolerate duplicate indices in this test
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)
        qkv = torch.randn((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)
        k = qkv[:, :HEAD_SIZE]
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]
        store_cache(
            k_cache,
            v_cache,
            indices,
            k,
            v,
        )

        # 2 = k + v
        MEM = bs * HEAD_SIZE * 2 * kv_cache.element_size()

        k = k.contiguous()
        v = v.contiguous()

        @torch.compile()
        def baseline():
            k_cache[indices] = k
            v_cache[indices] = v

        compare_memory_kernel_perf(
            our_impl=lambda: store_cache(k_cache, v_cache, indices, k, v),
            baseline=baseline,
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
