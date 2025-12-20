from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_p_values: torch.Tensor | None


class Sampler:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        if all(r.sampling_params.temperature <= 0.0 for r in batch.reqs):
            return BatchSamplingArgs(temperatures=None, top_p_values=None)

        MIN_T = 1e-5
        temperatures = torch.tensor(
            [max(r.sampling_params.temperature, MIN_T) for r in batch.reqs],
            dtype=torch.float32,
            pin_memory=True,
        ).to(self.device, non_blocking=True)

        # Only create top_p tensor if any request uses top_p < 1.0
        if all(r.sampling_params.top_p >= 1.0 for r in batch.reqs):
            top_p_values = None
        else:
            top_p_values = torch.tensor(
                [r.sampling_params.top_p for r in batch.reqs],
                dtype=torch.float32,
                pin_memory=True,
            ).to(self.device, non_blocking=True)

        return BatchSamplingArgs(temperatures=temperatures, top_p_values=top_p_values)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)
            return self._sample(logits, args.temperatures, args.top_p_values)

    def _sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p_values: torch.Tensor | None,
    ) -> torch.Tensor:
        logits.div_(temperatures.unsqueeze(-1))
        probs = torch.softmax(logits, dim=-1)

        if top_p_values is not None:
            probs = self._apply_top_p(probs, top_p_values)

        return torch.multinomial(probs, num_samples=1).view(-1)

    def _apply_top_p(self, probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Apply top-p (nucleus) sampling to probability distribution.

        Args:
            probs: (batch_size, vocab_size) probability distribution
            top_p: (batch_size,) top-p threshold for each sequence

        Returns:
            Modified probability distribution with top-p filtering applied
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # Keep at least one token (the first one)
        sorted_indices_to_remove = cumulative_probs > top_p.unsqueeze(-1)
        sorted_indices_to_remove[:, 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        probs.masked_fill_(indices_to_remove, 0.0)
        # Renormalize
        probs.div_(probs.sum(dim=-1, keepdim=True))
        return probs
