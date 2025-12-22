from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager
from .page_allocator import PageAllocator, pages_to_slots

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str, page_size: int):
        self.device = device
        self.allocator = PageAllocator(device, num_pages, page_size)
        self.manager = create_cache_manager(device=device, type=type, page_size=page_size)
        self.num_pages = num_pages
        self.page_size = page_size

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            pages = torch.unique(indices // self.page_size)
            self.allocator.free_pages(pages)

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return (self.manager.size_info.evictable_size + self.allocator.available_pages) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int, last_slot_idx: int = -1) -> torch.Tensor:
        """
        Allocate slots for tokens.
        last_slot_idx: The absolute slot index of the last allocated token for this request.
                       Used to identify and fill partial pages.
        """
        if needed_len == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        # Fill partial page if possible
        fill_slots = torch.empty(0, dtype=torch.int32, device=self.device)
        if last_slot_idx != -1:
            last_page_len = (last_slot_idx + 1) % self.page_size
            if last_page_len > 0: # If 0, it means the page was full
                fill_count = min(needed_len, self.page_size - last_page_len)
                fill_slots = torch.arange(last_slot_idx + 1, last_slot_idx + 1 + fill_count, 
                                          dtype=torch.int32, device=self.device)
                needed_len -= fill_count

        if needed_len == 0:
            return fill_slots

        # Allocate new pages for remainder
        num_pages_needed = (needed_len + self.page_size - 1) // self.page_size

        # Try to allocate from free pages
        pages = self.allocator.alloc_pages(num_pages_needed)

        if pages is None:
            needed_evict = num_pages_needed - self.allocator.available_pages
            evicted_pages = self.manager.evict(needed_evict)
            self.allocator.free_pages(evicted_pages)
            pages = self.allocator.alloc_pages(num_pages_needed)
            assert pages is not None, "Eviction did not free enough space."

        new_slots = pages_to_slots(pages, self.page_size, needed_len)
        return torch.cat([fill_slots, new_slots])

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[in_cache_len:])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        free_pages = self.allocator.available_pages        
        cached_info = self.manager.size_info
        cached_pages = cached_info.evictable_size + cached_info.protected_size
        total_pages = self.num_pages
        current_total = free_pages + cached_pages
        if current_total != total_pages:
            raise RuntimeError(
                f"CacheManager integrity check failed.\n"
                f"Expected Total Pages: {total_pages}\n"
                f"Actual Found: {current_total} (Free: {free_pages} + Cached: {cached_pages})\n"
                f"Difference: {total_pages - current_total}\n"
                "  - Positive difference implies memory LEAK (or active requests during check).\n"
                "  - Negative difference implies corruption (double free)."
            )
