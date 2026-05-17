from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool

if TYPE_CHECKING:
    from minisgl.hicache import HiCacheCounter


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
        layout: str,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self._device = device
        create_cache = lambda: _create_buffer(
            layout=layout,
            num_layers=num_layers,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=local_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self.k_buffer = create_cache()
        self.v_buffer = create_cache()
        self.storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self.counter: HiCacheCounter | None = None

    def k_cache(self, index: int) -> torch.Tensor:
        return self.k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.v_buffer[index]

    def set_hicache_counter(self, counter) -> None:
        self.counter = counter

    def get_kv_storage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_buffer, self.v_buffer

    def create_host_pool(self, num_pages: int, layout: str):
        num_layers, _, page_size, local_kv_heads, head_dim = self.k_buffer.shape
        return MHAKVCache(
            num_kv_heads=local_kv_heads * get_tp_info().size,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=self.dtype,
            device=torch.device("cpu"),
            layout=layout,
        )

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self.k_buffer[layer_id].view(self.storage_shape),
            v_cache=self.v_buffer[layer_id].view(self.storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
        if self.counter is not None:
            self.counter.wait(layer_id)

    def get_per_token_bytes(self) -> int:
        num_layers, _, _, local_kv_heads, head_dim = self.k_buffer.shape
        return 2 * num_layers * local_kv_heads * head_dim * self.k_buffer.element_size()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.k_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


def _create_buffer(
    layout: str,
    num_layers: int,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from minisgl.kernel import allocate_host

    ORDER_MAP = {
        "layer_first": (0, 1, 2, 3, 4),
        "page_first": (1, 2, 0, 3, 4),
    }
    shape = [num_layers, num_pages, page_size, num_kv_heads, head_dim]
    order = ORDER_MAP[layout]
    reverse_order = tuple(order.index(i) for i in range(len(order)))
    reordered_shape = [shape[i] for i in order]
    if device.type != "cpu":
        storage = torch.empty(reordered_shape, device=device, dtype=dtype)
    else:
        storage = allocate_host(*reordered_shape, dtype=dtype)
    return storage.permute(reverse_order)
