from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


class ForwardMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    IDLE = "idle"

    def is_prefill(self) -> bool:
        return self is ForwardMode.PREFILL

    def is_decode(self) -> bool:
        return self is ForwardMode.DECODE

    def is_extend(self) -> bool:
        # Keep naming aligned with sglang: both prefill/decode are extend-like.
        return self in (ForwardMode.PREFILL, ForwardMode.DECODE)

    def is_idle(self) -> bool:
        return self is ForwardMode.IDLE


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    batch: "Batch"
    reqs: list[Any]
    padded_reqs: list[Any]
    out_cache_loc: "torch.Tensor"
    num_token_non_padded_cpu: int
    attn_backend: Any
    out_cache_loc_swa: "torch.Tensor | None" = None

    @classmethod
    def from_batch(cls, batch: "Batch", *, attn_backend: Any) -> "ForwardBatch":
        return cls(
            forward_mode=ForwardMode(batch.phase),
            batch=batch,
            reqs=batch.reqs,
            padded_reqs=batch.padded_reqs,
            out_cache_loc=batch.out_loc,
            num_token_non_padded_cpu=sum(req.extend_len for req in batch.reqs),
            attn_backend=attn_backend,
            out_cache_loc_swa=None,
        )
