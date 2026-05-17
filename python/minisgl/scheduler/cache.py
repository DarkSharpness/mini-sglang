from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .config import SchedulerConfig
    from .utils import PendingReq


HICACHE_LOAD_LENGTH_THRESHOLD = 16


class CacheManager:
    def __init__(self, num_pages: int, page_table: torch.Tensor, config: SchedulerConfig):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        self.page_table = page_table
        self.page_size = page_size = config.page_size
        self.device = device = page_table.device
        self.num_pages = num_pages
        self.enable_hicache = config.cache_type == "hiradix"
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self._prefix_cache = create_prefix_cache(device=device, type=config.cache_type)
        if self.enable_hicache:
            from minisgl.hicache import HiCacheController

            self._hicache_controller = HiCacheController(self._prefix_cache, num_pages, config)
            self.start_load_host = self._hicache_controller.start_load
            self.refresh_hicache = self._hicache_controller.refresh

    def prepare_load_host(self, host_handle: BaseCacheHandle, cuda_handle: BaseCacheHandle):
        needed_len = host_handle.cached_len - cuda_handle.cached_len
        if needed_len < HICACHE_LOAD_LENGTH_THRESHOLD:
            return cuda_handle
        assert needed_len % self.page_size == 0, "allocation must be page-aliged"
        indices = self._page_to_token(self._allocate(needed_len // self.page_size))
        self._hicache_controller.prepare_load(host_handle, cuda_handle, indices)
        return host_handle

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self._prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self._prefix_cache.size_info.evictable_size + len(self._free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self._prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self._prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self._prefix_cache.insert_prefix(insert_ids, page_indices)
        if self.enable_hicache:
            self._hicache_controller.prepare_write(new_handle)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self._prefix_cache.check_integrity()
        cache_pages = self._prefix_cache.size_info.total_size // self.page_size
        if len(self._free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self._free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self._free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self._free_slots = torch.cat([self._free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self._free_slots)):
            evicted = self._prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self._free_slots = torch.cat([self._free_slots, evicted[:: self.page_size]])
            assert len(self._free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self._free_slots[:needed_pages]
        self._free_slots = self._free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
