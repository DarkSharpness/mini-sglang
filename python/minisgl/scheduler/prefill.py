from __future__ import annotations

from typing import TYPE_CHECKING, Final, List, Tuple

import torch
from minisgl.config.context import Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import SamplingParams, UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def __init__(
        self,
        *,
        chunk_size: int,
        host_ids: torch.Tensor,
        device_ids: torch.Tensor,
        table_idx: int,
        cached_len: int,
        output_len: int,
        uid: int,
        cache_handle: BaseCacheHandle,
        sampling_params: SamplingParams,
    ):
        assert host_ids.is_cpu
        self._real_output_len: Final = output_len
        self._host_ids: Final = host_ids  # full id
        self._device_ids: Final = device_ids  # pre-allocated device ids
        new_input_len = cached_len + chunk_size
        new_output_len = len(host_ids) + output_len - new_input_len
        assert new_input_len < len(host_ids) and chunk_size > 0
        assert len(device_ids) >= len(host_ids)
        super().__init__(
            input_ids=self._host_ids[:new_input_len],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=new_output_len,
            uid=uid,
            cache_handle=cache_handle,
            sampling_params=sampling_params,
        )

    def grow(self) -> None:
        self.cached_len = self.device_len

    @property
    def remain_chunk_size(self) -> int:
        return self.full_input_len - self.cached_len

    @property
    def full_input_len(self) -> int:
        return len(self._host_ids)

    def next_chunk(self, chunk_size: int) -> Req:
        new_input_len = self.cached_len + chunk_size
        assert self.cached_len == self.device_len and chunk_size > 0
        assert new_input_len <= self.full_input_len
        _slice = slice(self.cached_len, new_input_len)
        self._device_ids[_slice].copy_(self._host_ids[_slice].pin_memory(), non_blocking=True)
        self.host_ids = self._host_ids[:new_input_len]
        if new_input_len < self.full_input_len:
            return self
        return Req(
            input_ids=self._host_ids,
            table_idx=self.table_idx,
            cached_len=self.cached_len,
            output_len=self._real_output_len,
            uid=self.uid,
            cache_handle=self.cache_handle,
            sampling_params=self.sampling_params,
        )


class PrefillManager:
    def __init__(
        self,
        cache_manager: CacheManager,
        table_manager: TableManager,
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
                output_len=req.sampling_params.max_tokens,
            )
        )

    def try_allocate_one(
        self, req: PendingReq, inflight_tokens: int
    ) -> Tuple[BaseCacheHandle, int] | None:
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + inflight_tokens > self.cache_manager.available_size:
            return None

        self.cache_manager.lock(handle)
        if estimated_len + inflight_tokens > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        other_indices = self.cache_manager.allocate(extend_len)
        table_idx = self.table_manager.allocate()
        page_entry = self.table_manager.page_table[table_idx]

        # even for chunked req, we allocate the full extend_len here
        # if cached, copy the matched indices first
        if cached_len > 0:
            page_entry[:cached_len] = match_indices
        # since extend_len > 0, other indices must be non-empty
        page_entry[cached_len : cached_len + extend_len] = other_indices
        return handle, table_idx

    def schedule_next_batch(self, prefill_budget: int) -> Tuple[torch.Tensor, List[Req]] | None:
        from minisgl.kernel_v2 import make_2d_indices

        if len(self.pending_list) == 0:
            return None

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

            if not (resource := self.try_allocate_one(req, offset)):
                break

            handle, table_idx = resource
            cached_len = handle.cached_len
            extend_len = req.input_len - cached_len
            chunk_size = min(extend_len, prefill_budget)
            is_chunked = chunk_size < extend_len
            prefill_budget -= chunk_size

            # prepare token pool
            token_pool = self.table_manager.token_pool[table_idx]
            first_input_len = cached_len + chunk_size
            token_pool[:first_input_len].copy_(
                req.input_ids[:first_input_len].pin_memory(), non_blocking=True
            )
            if is_chunked:
                new_req = ChunkedReq(
                    chunk_size=chunk_size,
                    host_ids=req.input_ids,
                    device_ids=token_pool,
                    table_idx=table_idx,
                    cached_len=cached_len,
                    output_len=req.output_len,
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
                    table_idx=table_idx,
                    cached_len=cached_len,
                    uid=req.uid,
                    cache_handle=handle,
                    sampling_params=req.sampling_params,
                )
            result.append(new_req)

        if len(result) == 0:
            return None

        assert prefill_budget >= 0
        self.pending_list = chunked_list + self.pending_list[len(result) :]

        new_2d_indices = make_2d_indices(
            table_2d=self.table_manager.page_table,
            ranges=[(req.table_idx, req.cached_len, req.device_len) for req in result],
            load_table=False,
        )
        return new_2d_indices, result

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
