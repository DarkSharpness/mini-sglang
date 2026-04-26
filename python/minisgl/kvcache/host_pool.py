from __future__ import annotations

import threading
from typing import List

import torch


class HostKVCachePool:
    def __init__(
        self,
        device_cache: torch.Tensor,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, num_heads, head_dim),
            device="cpu",
            dtype=dtype,
            pin_memory=True,
        )
        self._num_layers = num_layers
        self._device_cache = device_cache
        self._lock = threading.RLock()
        self._free_pages = list(range(num_pages))
        self.num_pages = num_pages
        self.page_size = page_size

    @property
    def available_pages(self) -> int:
        return len(self._free_pages)

    def alloc(self, num_pages: int) -> torch.Tensor | None:
        with self._lock:
            if num_pages > len(self._free_pages):
                return None
            return torch.tensor(
                [self._free_pages.pop() for _ in range(num_pages)],
                dtype=torch.int32,
            )

    def free(self, indices: torch.Tensor) -> None:
        with self._lock:
            self._free_pages.extend(indices.tolist())

    def copy_from_device(self, device_indices: torch.Tensor, host_indices: torch.Tensor, stream: torch.cuda.Stream) -> None:
        device_pages = device_indices[:: self.page_size]
        host_pages = host_indices[:: self.page_size]
        for layer_id in range(self._num_layers):
            for d_page, h_page in zip(device_pages.tolist(), host_pages.tolist()):
                self._kv_buffer[0, layer_id, h_page].copy_(
                    self._device_cache[0, layer_id, d_page], non_blocking=True
                )
                self._kv_buffer[1, layer_id, h_page].copy_(
                    self._device_cache[1, layer_id, d_page], non_blocking=True
                )

    def copy_to_device(self, host_indices: torch.Tensor, device_indices: torch.Tensor, stream: torch.cuda.Stream) -> None:
        host_pages = host_indices[:: self.page_size]
        device_pages = device_indices[:: self.page_size]
        for layer_id in range(self._num_layers):
            for h_page, d_page in zip(host_pages.tolist(), device_pages.tolist()):
                self._device_cache[0, layer_id, d_page].copy_(
                    self._kv_buffer[0, layer_id, h_page], non_blocking=True
                )
                self._device_cache[1, layer_id, d_page].copy_(
                    self._kv_buffer[1, layer_id, h_page], non_blocking=True
                )
