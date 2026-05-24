from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .verify import verify_drafts


class SpeculativeEngine:
    """
    Single-request, greedy-verify speculative decoding on a HuggingFace backend.
    Matches greedy decoding that shares the same K-batch verify pass (e.g. HF
    assisted generation); bf16 near-ties may differ from per-token greedy. v1 is
    greedy-only and assumes full-attention models (rollback cannot crop a
    sliding-window cache).
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
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.draft = draft
        self.device = torch.device(device)
        self.target = AutoModelForCausalLM.from_pretrained(
            target_path, dtype=dtype
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)

        # Cumulative counters; call reset_stats() between independent runs.
        self.target_forward_ct = 0
        self.verify_ct = 0
        self.completion_tokens = 0
        self.num_correct_drafts = 0
        self.num_proposed_drafts = 0

    @property
    def accept_length(self) -> float:
        """τ (EAGLE): avg tokens per verify step, includes the bonus token."""
        if self.verify_ct == 0:
            return 0.0
        return self.completion_tokens / self.verify_ct

    @property
    def accept_rate(self) -> float:
        """α (Leviathan): per-draft-token acceptance probability, excludes bonus."""
        if self.num_proposed_drafts == 0:
            return 0.0
        return self.num_correct_drafts / self.num_proposed_drafts

    def reset_stats(self) -> None:
        self.target_forward_ct = 0
        self.verify_ct = 0
        self.completion_tokens = 0
        self.num_correct_drafts = 0
        self.num_proposed_drafts = 0

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
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        eos = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

        # Initial target prefill.
        input_ids = torch.tensor([prompt_ids], dtype=torch.int64, device=self.device)
        cache = DynamicCache(config=self.target.config)
        out = self.target(input_ids=input_ids, past_key_values=cache, use_cache=True)
        self.target_forward_ct += 1
        cache = out.past_key_values
        last_token = int(out.logits[0, -1].argmax().item())

        output_ids: list[int] = []
        if not _emit(output_ids, last_token, eos, max_new_tokens):
            self.completion_tokens += len(output_ids)
            return output_ids

        self.draft.warm_up(prompt_ids)

        while len(output_ids) < max_new_tokens:
            draft_tokens = self.draft.draft(last_token)
            if len(draft_tokens) != self.k:
                raise ValueError(
                    f"draft produced {len(draft_tokens)} tokens, expected k={self.k}"
                )

            verify_input = torch.tensor(
                [[last_token, *draft_tokens]], dtype=torch.int64, device=self.device
            )
            out = self.target(
                input_ids=verify_input, past_key_values=cache, use_cache=True
            )
            self.target_forward_ct += 1
            cache = out.past_key_values

            verify_logits = out.logits[0]  # (k+1, vocab)
            accept_tokens, num_correct_drafts = verify_drafts(
                torch.tensor(draft_tokens, dtype=torch.int64, device=self.device),
                verify_logits,
            )
            self.verify_ct += 1
            self.num_correct_drafts += num_correct_drafts
            self.num_proposed_drafts += self.k

            # Target cache truncation: discard entries for the rejected drafts.
            num_reject_drafts = self.k - num_correct_drafts
            cache.crop(cache.get_seq_length() - num_reject_drafts)

            self.draft.rollback(num_reject_drafts)

            stopped = False
            for tok in accept_tokens.tolist():
                if not _emit(output_ids, tok, eos, max_new_tokens):
                    stopped = True
                    break
            if stopped:
                break
            last_token = output_ids[-1]

        self.completion_tokens += len(output_ids)
        return output_ids


def _emit(output_ids: list[int], token: int, eos: int | None, limit: int) -> bool:
    """
    Append token to output_ids unless it is EOS or the limit is reached.
    Returns False if generation should stop after this call.
    """
    if eos is not None and token == eos:
        return False
    if len(output_ids) >= limit:
        return False
    output_ids.append(token)
    return len(output_ids) < limit
