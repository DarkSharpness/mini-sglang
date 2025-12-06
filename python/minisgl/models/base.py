from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.core import Batch, get_global_ctx
from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        ctx = get_global_ctx()
        with ctx.forward_batch(batch):
            return self.forward()
