from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import ReqConstraintState

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    constraint: ReqConstraintState | None = None
    chunked_req: ChunkedReq | None = None

    def __post_init__(self) -> None:
        if self.constraint is None and self.sampling_params.json_schema is not None:
            self.constraint = ReqConstraintState()

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def is_constrained(self) -> bool:
        return self.constraint is not None


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
