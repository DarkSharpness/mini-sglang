from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCache


@dataclass
class SamplingParams:
    top_k: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    max_tokens: int = 1024


class Req:
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,
        table_idx: int,
        cached_len: int,
        output_len: int,
        uid: int,
        sampling_params: SamplingParams,
        cache_handle: BaseCacheHandle,
    ) -> None:
        assert input_ids.is_cpu

        self.host_ids = input_ids
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = len(input_ids)
        self.max_device_len = len(input_ids) + output_len
        self.uid = uid
        self.sampling_params = sampling_params
        self.cache_handle = cache_handle

        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.host_ids = torch.cat([self.host_ids, next_token])

    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


class Batch:
    def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
        self.reqs = reqs
        self.phase: Literal["prefill", "decode"] = phase
        # these fields should be set by scheduler
        self.input_ids: torch.Tensor
        self.out_loc: torch.Tensor
        self.padded_reqs: List[Req]  # may contain some dummy reqs for padding
        # this field should be set by attention backend
        self.attn_metadata: BaseAttnMetadata

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


class Context:
    def __init__(
        self,
        *,
        page_size: int,
        kv_cache: BaseKVCache,
        attn_backend: BaseAttnBackend,
        page_table: torch.Tensor,
    ):
        self._batch: Batch | None = None
        self.page_table = page_table
        assert (
            self.page_table.dim() == 2
            and self.page_table.is_cuda
            and self.page_table.dtype == torch.int32
            and self.page_table.is_contiguous()
        )
        self.kv_cache = kv_cache
        self.attn_backend = attn_backend
        assert page_size == 1

    def set_batch(self, batch: Batch):
        assert self._batch is None
        self._batch = batch

    def reset_batch(self):
        assert self._batch is not None
        self._batch = None

    @contextmanager
    def forward_batch(self, batch: Batch):
        self.set_batch(batch)
        try:
            yield
        finally:
            self.reset_batch()

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "Global batch is not set"
        return self._batch


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
