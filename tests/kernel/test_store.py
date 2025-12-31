import pytest
import torch
from minisgl.kernel import store_cache


@pytest.mark.cuda
def test_store_cache_value_match(cuda_device):
    HEAD_SIZE = 128
    NUM_TOKENS = 1048576  # 1M
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device=cuda_device, dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    for bs in [2**n for n in range(0, 16)]:
        # NOTE: we cannot tolerate duplicate indices in this test
        indices = torch.randperm(NUM_TOKENS, device=cuda_device)[:bs].to(torch.int32)
        qkv = torch.randn((bs, HEAD_SIZE * 4), device=cuda_device, dtype=torch.float16)
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
