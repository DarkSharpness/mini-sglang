from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.kvcache.base import BaseKVCache, KVCacheLayout
from minisgl.utils import divide_even


class MHAKVCache(BaseKVCache):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in local LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        assert kv_layout != KVCacheLayout.MixPageLayer
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    @property
    def device(self) -> torch.device:
        return self._device
