from __future__ import annotations

from minisgl.benchmark.perf import compare_memory_kernel_perf
import torch
from minisgl.kernel import store_cache, store_mla_cache
from minisgl.utils import call_if_main


@call_if_main(__name__)
def test_store_cache():
    HEAD_SIZE = 128
    NUM_TOKENS = 1048576  # 1M
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    print("Testing Standard Cache Store...")
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

        assert torch.all(k_cache[indices] == k), bs
        assert torch.all(v_cache[indices] == v), bs

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


@call_if_main(__name__)
def test_store_mla_cache():
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    TOTAL_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    NUM_TOKENS = 1048576  # 1M
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_buffer = torch.randn((NUM_TOKENS, TOTAL_DIM), device="cuda", dtype=torch.float16)

    print("\nTesting MLA Cache Store...")
    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)
        kv_c = torch.randn((bs, KV_LORA_RANK), device="cuda", dtype=torch.float16)
        k_rope = torch.randn((bs, QK_ROPE_HEAD_DIM), device="cuda", dtype=torch.float16)
        store_mla_cache(
            kv_buffer,
            indices,
            kv_c,
            k_rope,
        )

        # Verify correct placement:
        # kv_c should be at [0 : KV_LORA_RANK]
        # k_rope should be at [KV_LORA_RANK : END]
        stored_c = kv_buffer[indices, :KV_LORA_RANK]
        stored_rope = kv_buffer[indices, KV_LORA_RANK:]
        assert torch.all(stored_c == kv_c), f"KV_C mismatch at BS={bs}"
        assert torch.all(stored_rope == k_rope), f"K_ROPE mismatch at BS={bs}"

        MEM = bs * TOTAL_DIM * kv_buffer.element_size()

        kv_c = kv_c.contiguous()
        k_rope = k_rope.contiguous()

        @torch.compile()
        def baseline():
            kv_buffer[indices, :KV_LORA_RANK] = kv_c
            kv_buffer[indices, KV_LORA_RANK:] = k_rope

        compare_memory_kernel_perf(
            our_impl=lambda: store_mla_cache(kv_buffer, indices, kv_c, k_rope),
            baseline=baseline,
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
