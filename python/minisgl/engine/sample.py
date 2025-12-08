from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None


class Sampler:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        if all(r.sampling_params.temperature <= 0.0 for r in batch.reqs):
            return BatchSamplingArgs(temperatures=None)
        MIN_T = 1e-5
        return BatchSamplingArgs(
            temperatures=torch.tensor(
                [max(r.sampling_params.temperature, MIN_T) for r in batch.reqs],
                dtype=torch.float32,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)
            return self._sample(logits, args.temperatures)

    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits.div_(temperatures.unsqueeze(-1))
        torch.softmax(logits, dim=-1, out=logits)
        return torch.multinomial(logits, num_samples=1).view(-1)
