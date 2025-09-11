from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCache


class Req:
    def __init__(
        self,
        *,
        input_ids: List[int] | torch.Tensor,
        page_table_idx: int,
        cached_len: int,
        output_len: int,
        device: torch.device,
        uid: int,
        cache_handle: BaseCacheHandle | None = None,
    ):
        input_ids = (
            (input_ids.pin_memory() if not input_ids.is_cuda else input_ids)
            if isinstance(input_ids, torch.Tensor)
            else torch.tensor(input_ids, dtype=torch.int32, pin_memory=True)
        )
        if not input_ids.is_cuda:
            input_ids = input_ids.to(device, non_blocking=True)
        self.device_ids = input_ids
        self.page_table_idx = page_table_idx
        self.cached_len = cached_len
        self.max_device_len = len(input_ids) + output_len
        self.uid = uid

        assert 0 <= self.cached_len < self.device_len

        # this field should be set by scheduler
        if cache_handle is not None:
            self.cache_handle = cache_handle

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return len(self.device_ids) - self.cached_len

    @property
    def device_len(self) -> int:
        return len(self.device_ids)

    def append(self, next_token: torch.Tensor) -> None:
        """
        Append a token to the request. The token must be a 1-D tensor with shape (1,).

        Note that for subclass of Req (e.g. Chunked Prefill Req), this method can
        be overridden to handle more complex logic.
        """
        self.device_ids = torch.cat([self.device_ids, next_token], dim=0)
        self.cached_len = len(self.device_ids) - 1

    def __repr__(self) -> str:
        return (
            f"{type(self)}(page_table_idx={self.page_table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Phase:
    _phase: Literal["prefill", "decode"]

    @property
    def is_prefill(self) -> bool:
        return self._phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self._phase == "decode"


class Batch(Phase):
    @staticmethod
    def _auto_phase(reqs: List[Req], hint: Literal["prefill", "decode"] | None):
        if hint is not None:
            result = Batch._auto_phase(reqs, None)
            assert result == hint, f"Phase hint {hint} conflicts with reqs"
            return hint
        if all(req.extend_len == 1 for req in reqs):
            return "decode"
        else:
            return "prefill"

    def __init__(
        self,
        *,
        reqs: List[Req],
        phase: Literal["prefill", "decode"] | None = None,
    ):
        self.reqs = reqs
        super().__init__(_phase=self._auto_phase(reqs, phase))
        # these field will be set later by attention backend
        self.attn_metadata: BaseAttnMetadata
        self.input_ids: torch.Tensor
        # only not equal to batch_size when this batch uses CUDA graph
        self.padded_bs: int

    @property
    def batch_size(self) -> int:
        return len(self.reqs)


class Context:
    def __init__(
        self,
        *,
        page_num: int,
        page_size: int,
        max_running_req: int,
        max_seq_len: int,
        device: torch.device,
        kv_cache: BaseKVCache,
        attn_backend: BaseAttnBackend,
    ):
        self._batch: Batch | None = None
        self.page_table = torch.randint(
            low=0,
            high=max(page_num, 1),
            size=(max_running_req, max_seq_len),
            dtype=torch.int32,
            device=device,
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


__all__ = [
    "Context",
    "set_global_ctx",
    "get_global_ctx",
    "Req",
    "Batch",
    "Phase",
]
