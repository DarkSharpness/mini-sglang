from __future__ import annotations

import torch

import minisgl.distributed.info as _dist_info
from minisgl.kvcache.quantized_mha_pool import QuantizedMHAKVCache, _FP8_E4M3FN_MAX
from minisgl.utils import call_if_main

_FP8 = torch.float8_e4m3fn


def _make_pool(num_pages=16, page_size=1, num_heads=4, head_dim=64, num_layers=1):
    _dist_info._TP_INFO = None
    _dist_info.set_tp_info(rank=0, size=1)
    return QuantizedMHAKVCache(
        num_kv_heads=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )


def _flat(t):
    return t.reshape(t.shape[0], -1).contiguous()


def _read(pool, indices, layer=0, side="k"):
    buf = pool.k_cache(layer) if side == "k" else pool.v_cache(layer)
    return buf.reshape(-1, buf.shape[-2], buf.shape[-1])[indices]


@call_if_main(__name__)
def test_quantized_kvcache():
    pool = _make_pool()
    assert pool.dtype == torch.float16
    assert pool.store_dtype == _FP8

    indices = torch.tensor([1, 5, 11], device="cuda", dtype=torch.int64)
    torch.manual_seed(0)

    # in-range round-trip + clamp saturation + NaN/Inf propagation, byte-exact
    k = torch.randn(3, 4, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(3, 4, 64, dtype=torch.float16, device="cuda")
    k[0, 0, 0] = 1000.0       # saturates to +448
    v[0, 0, 0] = -1000.0      # saturates to -448
    k[1, 0, 0] = float("nan") # NaN -> NaN
    v[1, 0, 0] = float("inf") # +inf -> +448 (via clamp)

    expected_k = k.clamp(-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX).to(_FP8)
    expected_v = v.clamp(-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX).to(_FP8)
    pool.store_kv(_flat(k), _flat(v), indices, layer_id=0)

    got_k = _read(pool, indices, side="k")
    got_v = _read(pool, indices, side="v")
    assert torch.equal(got_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(got_v.view(torch.uint8), expected_v.view(torch.uint8))

    k_fp16 = got_k.to(torch.float16)
    v_fp16 = got_v.to(torch.float16)
    assert k_fp16[0, 0, 0].item() == _FP8_E4M3FN_MAX
    assert v_fp16[0, 0, 0].item() == -_FP8_E4M3FN_MAX
    assert torch.isnan(k_fp16[1, 0, 0]).item()
    assert v_fp16[1, 0, 0].item() == _FP8_E4M3FN_MAX  # inf clamped


@call_if_main(__name__)
def test_default_cast_to_fp8_produces_nan():
    """Pins Phase 0 (A1): plain .to(fp8) does NOT saturate -- out-of-range -> NaN.

    If a future torch makes .to(fp8) saturating, remove the clamp in
    QuantizedMHAKVCache.store_kv.
    """
    x = torch.tensor([500.0], dtype=torch.float16, device="cuda")
    assert torch.isnan(x.to(_FP8).to(torch.float16)).item()
