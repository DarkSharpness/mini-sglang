from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self.free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def lock_guard(self, handle: BaseCacheHandle):
        return self.manager.lock_handle_guard(handle)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        if needed_len <= (free_len := len(self.free_slots)):
            allocated = self.free_slots[:needed_len]
            self.free_slots = self.free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        if needed_len <= len(evicted):
            allocated = evicted[:needed_len]
            self.free_slots = torch.cat([self.free_slots, evicted[needed_len:]])
            return allocated

        merged = torch.cat([self.free_slots, evicted])
        merged_len = len(merged)
        assert merged_len >= needed_len, "Eviction did not free enough space."

        if merged_len == needed_len:
            self.free_slots = torch.empty(0, dtype=torch.int32, device=self.device)
            return merged
        else:
            allocated = merged[:needed_len]
            self.free_slots = merged[needed_len:]
            return allocated

    def free(self, handle: BaseCacheHandle, input_ids: torch.Tensor, indices: torch.Tensor) -> None:
        assert input_ids.is_cpu, "input_ids should not be on GPU"
        self.unlock(handle)
        old_cache_len = handle.cached_len
        new_cache_len = self.manager.insert_prefix(input_ids, indices)
        # these indices are already in cache (possibly cached by other reqs), so we need to free them
        self.free_slots = torch.cat([self.free_slots, indices[old_cache_len:new_cache_len]])

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self.free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self.free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
