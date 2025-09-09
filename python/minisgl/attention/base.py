from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.config.context import Batch, Req


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def finalize(self, page_table: torch.Tensor) -> None: ...
    @abstractmethod
    def get_positions(self) -> torch.Tensor: ...
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, scale: float
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch, allow_graph: bool) -> bool: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], dummy_req: Req) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...
