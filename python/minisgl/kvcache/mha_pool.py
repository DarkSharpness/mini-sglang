from __future__ import annotations

from typing import Literal

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
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
        layout: Literal["nhd", "bnbsd"] = "nhd",
    ) -> None:
        if layout not in ("nhd", "bnbsd"):
            raise ValueError(
                f"MHAKVCache layout must be 'nhd' or 'bnbsd', got {layout!r}"
            )
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        if layout == "nhd":
            self._kv_buffer = torch.empty(
                (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
                device=device,
                dtype=dtype,
            )
        else:  # bnbsd — FIA paged-KV native layout on Ascend
            self._kv_buffer = torch.empty(
                (2, num_layers, num_pages, local_kv_heads, page_size, head_dim),
                device=device,
                dtype=dtype,
            )
        self._layout = layout
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        if self._layout == "nhd":
            from minisgl.kernel import store_cache

            store_cache(
                k_cache=self._k_buffer[layer_id].view(self._storage_shape),
                v_cache=self._v_buffer[layer_id].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
        else:  # bnbsd — pure PyTorch scatter, no CUDA kernel dependency
            page_ids = out_loc // self._page_size
            offsets = out_loc % self._page_size
            k_values = k.view(-1, self._local_kv_heads, self._head_dim)
            v_values = v.view(-1, self._local_kv_heads, self._head_dim)
            self._k_buffer[layer_id][page_ids, :, offsets, :] = k_values
            self._v_buffer[layer_id][page_ids, :, offsets, :] = v_values

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
