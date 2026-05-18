"""Vanilla speculative-decoding engine for a single request.

Uses HuggingFace transformers + ``DynamicCache`` as the inference backend. The
mini-sglang touchpoint is :func:`verify_drafts` from :mod:`.verify`; everything
else here is generic Python+HF plumbing. Integration with mini-sglang's own
Engine/Scheduler is deferred to a follow-up PR (see ``phase_0_spike_report.md``).
"""
from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .verify import verify_drafts


class SpeculativeEngine:
    """Single-request, greedy-verify speculative decoding.

    The target's output is guaranteed token-for-token identical to non-speculative
    greedy decoding on the same prompt (the project's non-negotiable correctness
    invariant). Sampling other than greedy is intentionally unsupported in v1.
    """

    def __init__(
        self,
        target_path: str,
        draft,
        k: int = 4,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        self.k = k
        self.draft = draft
        self.device = torch.device(device)
        self.target = AutoModelForCausalLM.from_pretrained(
            target_path, dtype=dtype
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)

        # Benchmark counters (Phase 4 will consume these).
        self.target_forward_calls = 0
        self.tokens_generated = 0
        self.accepted_lengths: list[int] = []

    @property
    def average_acceptance_length(self) -> float:
        if not self.accepted_lengths:
            return 0.0
        return sum(self.accepted_lengths) / len(self.accepted_lengths)

    def reset_stats(self) -> None:
        self.target_forward_calls = 0
        self.tokens_generated = 0
        self.accepted_lengths.clear()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | Iterable[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[int]:
        prompt_ids = (
            self.tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        eos = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

        # Initial target prefill.
        input_ids = torch.tensor([prompt_ids], dtype=torch.int64, device=self.device)
        cache = DynamicCache()
        out = self.target(input_ids=input_ids, past_key_values=cache, use_cache=True)
        self.target_forward_calls += 1
        cache = out.past_key_values
        t_last = int(out.logits[0, -1].argmax().item())

        output_ids: list[int] = []
        if not _emit(output_ids, t_last, eos, max_new_tokens):
            self.tokens_generated = len(output_ids)
            return output_ids

        self.draft.warm_up(prompt_ids)

        while len(output_ids) < max_new_tokens:
            drafts = self.draft.draft(t_last, self.k)
            assert len(drafts) == self.k
            drafts_tensor = torch.tensor(drafts, dtype=torch.int64, device=self.device)

            verify_input = torch.tensor(
                [[t_last, *drafts]], dtype=torch.int64, device=self.device
            )
            out = self.target(
                input_ids=verify_input, past_key_values=cache, use_cache=True
            )
            self.target_forward_calls += 1
            cache = out.past_key_values

            verify_logits = out.logits[0]  # (k+1, vocab)
            accepted_tokens, accepted_drafts = verify_drafts(
                drafts_tensor, verify_logits
            )
            self.accepted_lengths.append(accepted_drafts)

            # Target cache truncation: discard entries for the rejected drafts.
            rejected = self.k - accepted_drafts
            cache.crop(cache.get_seq_length() - rejected)

            self.draft.rollback(rejected)

            stopped = False
            for tok in accepted_tokens.tolist():
                if not _emit(output_ids, tok, eos, max_new_tokens):
                    stopped = True
                    break
            if stopped:
                break
            t_last = output_ids[-1]

        self.tokens_generated = len(output_ids)
        return output_ids


def _emit(output_ids: list[int], token: int, eos: int | None, limit: int) -> bool:
    """Append ``token`` to ``output_ids`` unless it's EOS or we hit the limit.

    Returns False if generation should stop after this call. Matches
    ``LLM.offline_send_result`` semantics: an EOS terminates without being
    emitted (only when EOS is not being ignored at the caller's request).
    """
    if eos is not None and token == eos:
        return False
    if len(output_ids) >= limit:
        return False
    output_ids.append(token)
    return len(output_ids) < limit
