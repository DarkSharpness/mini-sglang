from __future__ import annotations

import torch

from .mha_pool import MHAKVCache

_FP8_E4M3FN_MAX = 448.0


class QuantizedMHAKVCache(MHAKVCache):
    """MHA KV cache stored as float8_e4m3fn with implicit scale=1.0.

    Calibrated k_scale/v_scale from W8A8 checkpoints are silently ignored.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self._compute_dtype = dtype

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # torch.to(float8_e4m3fn) does NOT saturate -- out-of-range fp16 lands as NaN.
        k_q = k.clamp(-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
        v_q = v.clamp(-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
        super().store_kv(k_q, v_q, out_loc, layer_id)

    @property
    def dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def store_dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype
