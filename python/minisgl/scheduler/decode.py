from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Set

if TYPE_CHECKING:
    from minisgl.config.context import Req

    from .cache import CacheManager
    from .table import PageTableManager


class DecodeManager:
    def __init__(self, cache_manager: CacheManager, table_manager: PageTableManager):
        self.running_reqs: Set[Req] = set()
        self.cache_manager = cache_manager
        self.table_manager = table_manager

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.update(reqs)

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def remove_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.difference_update(reqs)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> List[Req] | None:
        if len(self.running_reqs) == 0:
            return None

        from minisgl.kernel_v2 import store_decode_indices

        decode_bs = len(self.running_reqs)
        if self.cache_manager.available_size < decode_bs:
            raise NotImplementedError("TODO: Implement decode retract")

        reqs = list(self.running_reqs)
        out_loc = self.cache_manager.allocate(decode_bs)

        store_decode_indices(
            page_table=self.table_manager.page_table,
            indices=out_loc,
            pos=[(req.page_table_idx, req.cached_len) for req in reqs],
        )

        return reqs

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
