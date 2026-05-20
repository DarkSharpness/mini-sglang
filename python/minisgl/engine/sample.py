from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.constrained import BaseGrammarObject
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    grammars: List[BaseGrammarObject | None] | None = None

    @property
    def has_grammar(self) -> bool:
        return self.grammars is not None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def _apply_grammar_mask(self, logits: torch.Tensor, args: BatchSamplingArgs) -> None:
        grammars = args.grammars
        if grammars is None:
            return

        first_grammar = next((grammar for grammar in grammars if grammar is not None), None)
        if first_grammar is None:
            return

        vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(grammars),
            device=logits.device,
        )
        for i, grammar in enumerate(grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(vocab_mask, i)
        vocab_mask = first_grammar.move_vocab_mask(vocab_mask, logits.device)
        first_grammar.apply_vocab_mask(logits, vocab_mask)

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        grammars = None
        if any(r.is_constrained for r in batch.reqs):
            grammars = [r.constraint.grammar if r.constraint else None for r in batch.reqs]

        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None, grammars=grammars)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, grammars=grammars)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            self._apply_grammar_mask(logits, args)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
