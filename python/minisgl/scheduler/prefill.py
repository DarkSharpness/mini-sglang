from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.config.context import Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import SamplingParams, UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import PageTableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def __init__(
        self,
        *,
        chunk_size: int,
        full_input_ids: torch.Tensor,
        page_table_idx: int,
        cached_len: int,
        output_len: int,
        device: torch.device,
        uid: int,
        cache_handle: BaseCacheHandle,
        sampling_params: SamplingParams,
    ):
        assert full_input_ids.is_cpu
        self._input_ids_cpu: Final = full_input_ids
        self._real_output_len: Final = output_len
        new_input_len = cached_len + chunk_size
        new_output_len = len(full_input_ids) + output_len - new_input_len
        assert new_input_len < len(full_input_ids) and chunk_size > 0
        super().__init__(
            input_ids=self._input_ids_cpu[:new_input_len],
            page_table_idx=page_table_idx,
            cached_len=cached_len,
            output_len=new_output_len,
            device=device,
            uid=uid,
            cache_handle=cache_handle,
            sampling_params=sampling_params,
        )

    def append(self, next_token: torch.Tensor) -> None:
        self.cached_len = len(self.device_ids)
        _ = next_token  # unused, because this is chunked req

    @property
    def remain_chunk_size(self) -> int:
        return len(self._input_ids_cpu) - self.cached_len

    def next_chunk(self, chunk_size: int) -> Req:
        new_input_len = self.cached_len + chunk_size
        assert self.cached_len == len(self.device_ids) and chunk_size > 0
        assert new_input_len <= len(self._input_ids_cpu)
        extra_ids = self._input_ids_cpu[self.cached_len : new_input_len]
        device = self.device_ids.device
        extra_ids = extra_ids.pin_memory().to(device, non_blocking=True)
        self.device_ids = torch.cat([self.device_ids, extra_ids], dim=0)
        if new_input_len < len(self._input_ids_cpu):
            return self
        return Req(
            input_ids=self.device_ids,
            page_table_idx=self.page_table_idx,
            cached_len=self.cached_len,
            output_len=self._real_output_len,
            device=device,
            uid=self.uid,
            cache_handle=self.cache_handle,
            host_ids=self._input_ids_cpu,
            sampling_params=self.sampling_params,
        )


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
                sampling_params=req.sampling_params,
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
        chunked_list: List[PendingReq] = []
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

            # if this is a chunked req, continue the chunking or finish it
            if (chunked_req := req.chunked_req) is not None:
                req.chunked_req = None
                chunk_size = min(chunked_req.remain_chunk_size, prefill_budget)
                assert chunk_size > 0
                prefill_budget -= chunk_size
                new_req = chunked_req.next_chunk(chunk_size)
                if isinstance(new_req, ChunkedReq):
                    req.chunked_req = new_req
                    chunked_list.append(req)
                result.append(new_req)
                continue

            handle, match_indices = self.cache_manager.match_req(req)
            cached_len = handle.cached_len

            # TODO: better estimate policy
            extend_len = req.input_len - cached_len
            estimated_len = extend_len + req.output_len

            chunk_size, is_chunked = extend_len, False
            if extend_len > prefill_budget:
                chunk_size, is_chunked = prefill_budget, True

            # NOTE: early reject here to avoid unnecessary lock
            if estimated_len > self.cache_manager.available_size - offset:
                break

            # NOTE: drop the lock if we cannot allocate
            self.cache_manager.lock(handle)
            if estimated_len > self.cache_manager.available_size - offset:
                self.cache_manager.unlock(handle)
                break

            # NOTE: once we reach here, we must be able to allocate
            assert chunk_size > 0 and extend_len > 0
            prefill_budget -= chunk_size

            # even for chunked req, we allocate the full extend_len here
            other_indices = self.cache_manager.allocate(extend_len)
            page_idx = self.table_manager.allocate()
            page_entry = page_table[page_idx]
            # if cached, copy the matched indices first
            if cached_len > 0:
                page_entry[:cached_len] = match_indices
            # since extend_len > 0, other indices must be non-empty
            page_entry[cached_len : cached_len + extend_len] = other_indices

            if is_chunked:
                new_req = ChunkedReq(
                    chunk_size=chunk_size,
                    full_input_ids=req.input_ids,
                    page_table_idx=page_idx,
                    cached_len=cached_len,
                    output_len=req.output_len,
                    device=self.cache_manager.device,
                    uid=req.uid,
                    cache_handle=handle,
                    sampling_params=req.sampling_params,
                )
                req.chunked_req = new_req
                chunked_list.append(req)
            else:
                new_req = Req(
                    input_ids=req.input_ids,
                    output_len=req.output_len,
                    page_table_idx=page_idx,
                    cached_len=cached_len,
                    device=self.cache_manager.device,
                    uid=req.uid,
                    cache_handle=handle,
                    sampling_params=req.sampling_params,
                )
            result.append(new_req)

        if len(result) == 0:
            return None

        assert prefill_budget >= 0
        self.pending_list = chunked_list + self.pending_list[len(result) :]
        return result

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
