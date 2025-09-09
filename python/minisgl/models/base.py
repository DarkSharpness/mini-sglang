from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.config.context import Batch, get_global_ctx

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        ctx = get_global_ctx()
        with ctx.forward_batch(batch):
            return self.forward()
