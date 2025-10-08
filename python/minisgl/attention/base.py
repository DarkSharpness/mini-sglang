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
    def get_positions(self) -> torch.Tensor: ...
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch, allow_graph: bool) -> None:
        """Prepare metadata for the current batch.

        Args:
            batch (Batch): The current batch.
            allow_graph (bool): Whether to allow CUDA graph capture.
        """

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], dummy_req: Req) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, bs: int) -> Batch: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        if batch.is_prefill:
            return self.prefill_backend.forward(q, k, v, layer_id, batch)
        else:
            return self.decode_backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch, allow_graph: bool) -> None:
        if batch.is_prefill:
            self.prefill_backend.prepare_metadata(batch, allow_graph)
        else:
            self.decode_backend.prepare_metadata(batch, allow_graph)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], dummy_req: Req) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list, dummy_req)

    def prepare_for_capture(self, bs: int) -> Batch:
        return self.decode_backend.prepare_for_capture(bs)

    def prepare_for_replay(self, batch: Batch) -> None:
        assert batch.is_decode
        self.decode_backend.prepare_for_replay(batch)
