from typing import Callable
import torch
from minisgl.kernel_v2 import store_cache
from minisgl.utils import call_if_main


def perf_func(f: Callable):
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(100):
            f()
    torch.cuda.synchronize()

    g.replay()
    tic.record()
    g.replay()
    toc.record()
    toc.synchronize()
    dur = tic.elapsed_time(toc)
    return dur / 100.0


@call_if_main(__name__)
def test_store_cache():
    HEAD_SIZE = 128 * 8
    NUM_TOKENS = 200000
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)
        qkv = torch.empty((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)
        k = qkv[:, :HEAD_SIZE]
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]
        store_cache(
            k_cache,
            v_cache,
            indices,
            k,
            v,
        )
        assert torch.all(k_cache[indices] == k)
        assert torch.all(v_cache[indices] == v)

        # test the perf
        dur = perf_func(lambda: store_cache(k_cache, v_cache, indices, k, v))
        bandwidth = (bs * HEAD_SIZE * 2 * 2) / (dur * 1e6)  # GB/s
        print(f"BS={bs:6d} | Our Store: {dur:8.3f} ms | {bandwidth:8.3f} GB/s")

        # k = k.contiguous()
        # v = v.contiguous()

        def baseline():
            k_cache[indices] = k
            v_cache[indices] = v

        dur = perf_func(baseline)
        bandwidth = (bs * HEAD_SIZE * 2 * 2) / (dur * 1e6)  # GB/s
        print(f"BS={bs:6d} | Baseline : {dur:8.3f} ms | {bandwidth:8.3f} GB/s")
