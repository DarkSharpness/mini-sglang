from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Set, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.core import Req

    from .cache import CacheManager
    from .table import TableManager


class DecodeManager:
    def __init__(self, cache_manager: CacheManager, table_manager: TableManager):
        self.running_reqs: Set[Req] = set()
        self.cache_manager = cache_manager
        self.table_manager = table_manager

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def remove_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.difference_update(reqs)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Tuple[torch.Tensor, List[Req]] | None:
        from minisgl.kernel_v2 import make_2d_indices

        if len(self.running_reqs) == 0:
            return None

        decode_bs = len(self.running_reqs)
        if self.cache_manager.available_size < decode_bs:
            raise NotImplementedError("TODO: Implement decode retract")

        reqs = list(self.running_reqs)
        new_2d_indices = make_2d_indices(
            table_2d=self.table_manager.page_table,
            ranges=[(req.table_idx, req.cached_len, req.device_len) for req in reqs],
            load_table=False,
            store_value=self.cache_manager.allocate(decode_bs),
        )

        return new_2d_indices, reqs

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
