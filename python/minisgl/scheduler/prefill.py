from __future__ import annotations

from typing import TYPE_CHECKING, List

from minisgl.config.context import Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import PageTableManager

logger = init_logger(__name__)


class PrefillManager:
    def __init__(
        self,
        cache_manager: CacheManager,
        table_manager: PageTableManager,
        decode_manager: DecodeManager,
    ) -> None:
        self.pending_list: List[PendingReq] = []
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.decode_manager = decode_manager

    def add_raw_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                uid=req.uid,
                input_ids=req.input_ids,
                output_len=req.output_len,
            )
        )

    def schedule_next_batch(self, prefill_budget: int) -> List[Req] | None:
        if len(self.pending_list) == 0:
            return None

        page_table = self.table_manager.page_table
        # estimated offset due to in-flight decode
        offset = self.decode_manager.inflight_tokens

        # use FIFO
        result: List[Req] = []
        for req in self.pending_list:
            # We need at least 1 prefill budget, 1 table entry, and 1 cache space
            if (
                min(
                    prefill_budget,
                    self.cache_manager.available_size - offset,
                    self.table_manager.available_size,
                )
                <= 0
            ):
                break
            handle, match_indices = self.cache_manager.match_req(req)
            cached_len = len(match_indices)

            # TODO: better estimate policy
            extend_len = req.input_len - cached_len
            estimated_len = extend_len + req.output_len

            # TODO: handle this with chunked prefill
            if extend_len > prefill_budget:
                break

            # NOTE: early reject here to avoid unnecessary lock
            if estimated_len > self.cache_manager.available_size - offset:
                break

            # NOTE: drop the lock if we cannot allocate
            self.cache_manager.lock(handle)
            if estimated_len > self.cache_manager.available_size - offset:
                self.cache_manager.unlock(handle)
                break

            assert extend_len > 0
            other_indices = self.cache_manager.allocate(extend_len)
            page_idx = self.table_manager.allocate()
            prefill_budget -= extend_len
            page_entry = page_table[page_idx]

            if cached_len > 0:
                page_entry[:cached_len] = match_indices
            if extend_len > 0:
                page_entry[cached_len : req.input_len] = other_indices

            result.append(
                Req(
                    input_ids=req.input_ids,
                    output_len=req.output_len,
                    page_table_idx=page_idx,
                    cached_len=cached_len,
                    device=self.cache_manager.device,
                    uid=req.uid,
                    cache_handle=handle,
                ),
            )

        if len(result) == 0:
            return None

        assert prefill_budget >= 0
        self.pending_list = self.pending_list[len(result) :]
        return result

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
