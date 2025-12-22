"""
Page-level memory allocator
"""
from __future__ import annotations

import torch
from dataclasses import dataclass

@dataclass
class PageAllocator:
    device: torch.device
    num_pages: int
    page_size: int

    def __post_init__(self):
        self._free_pages = torch.arange(self.num_pages, dtype=torch.int32, device=self.device)
        self._is_page_free = torch.ones(self.num_pages, dtype=torch.bool, device=self.device)

    def alloc_pages(self, num_pages: int) -> torch.Tensor | None:
        if num_pages > len(self._free_pages):
            return None
        allocated = self._free_pages[:num_pages]
        self._free_pages = self._free_pages[num_pages:]
        self._is_page_free[allocated] = False
        return allocated

    def free_pages(self, page_indices: torch.Tensor) -> None:
        if len(page_indices) == 0:
            return
        self._is_page_free[page_indices] = True
        self._free_pages = torch.cat([self._free_pages, page_indices])
        # Keep sorted for determinism (good for debugging)
        self._free_pages, _ = torch.sort(self._free_pages)

    @property
    def available_pages(self) -> int:
        return len(self._free_pages)


def pages_to_slots(page_indices: torch.Tensor, page_size: int,
                   needed_slots: int, start_offset: int = 0) -> torch.Tensor:
    if len(page_indices) == 0:
        return torch.empty(0, dtype=torch.int32, device=page_indices.device)
    page_starts = page_indices * page_size
    offsets = torch.arange(page_size, device=page_indices.device)
    all_slots = (page_starts.unsqueeze(1) + offsets.unsqueeze(0)).view(-1)
    return all_slots[start_offset : start_offset + needed_slots]
