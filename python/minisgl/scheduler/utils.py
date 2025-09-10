from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    output_len: int

    @property
    def input_len(self) -> int:
        return len(self.input_ids)


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
