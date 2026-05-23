"""Draft strategies for speculative decoding.

A draft instance must provide three methods (no `Protocol` declared in v1 — the
contract is enforced by duck typing; extract a protocol when the third strategy
arrives):

- ``warm_up(prompt_ids)``: feed the prompt so the draft's internal state mirrors
  the target's after its initial prefill. May be a no-op.
- ``draft(last_token)``: autoregressively generate ``self.k`` candidate token
  IDs starting from ``last_token``. Returns a Python list of ``self.k`` ints.
  After a draft round the draft's internal state should reflect K+1 advanced
  positions (KV for ``[last_token, d1, ..., dK]``) so it lines up 1:1 with the
  target's K+1-wide verify pass.
- ``rollback(num_reject_drafts)``: discard the trailing ``num_reject_drafts``
  entries from the draft's internal state. Symmetric with the target engine's
  ``cache.crop(cache.get_seq_length() - num_reject_drafts)``. May be a no-op.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, DynamicCache


class DummyDraft:
    """Always returns the same wrong token. Used to exercise the rejection path."""

    def __init__(self, k: int, wrong_token: int = 100_000) -> None:
        self.k = k
        self.wrong_token = wrong_token

    def warm_up(self, prompt_ids: list[int]) -> None:
        pass

    def draft(self, last_token: int) -> list[int]:
        return [self.wrong_token] * self.k

    def rollback(self, num_reject_drafts: int) -> None:
        pass


class StandaloneDraft:
    """A real autoregressive draft using a smaller standalone model from the target's family.

    "Standalone" follows SGLang's terminology for a draft strategy backed by a
    separate smaller LLM (as opposed to EAGLE/MTP feature-based heads). The
    speculative *algorithm* here is still vanilla single-chain greedy — this
    names only the draft type.

    Per draft round the cache is advanced by K+1 positions: K forwards generate
    the K candidate tokens, then one extra forward feeds the K-th candidate so
    its KV lands in the cache. This keeps the draft cache aligned 1:1 with the
    target's K+1-wide verify pass (which holds KV for ``[last_token, d1, ..., dK]``),
    so ``rollback(num_reject_drafts)`` is a plain truncation by ``num_reject_drafts``
    with no off-by-one — matching ``DynamicCache.crop`` on the target side.

    The alternative (K forwards, no trailing extra) leaves the draft cache one
    position behind the target after an all-accept round; the next round's
    draft proposals are then conditioned on a wrong-by-one prefix and silently
    lose acceptance rate. The extra forward costs ~20% of draft compute at K=4,
    well below the target's per-round cost.
    """

    def __init__(
        self,
        draft_path: str,
        k: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        self.k = k
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            draft_path, dtype=dtype
        ).to(self.device).eval()
        self.cache: DynamicCache = DynamicCache(config=self.model.config)

    def _step(self, input_ids: torch.Tensor):
        out = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out

    @torch.inference_mode()
    def warm_up(self, prompt_ids: list[int]) -> None:
        self.cache = DynamicCache(config=self.model.config)
        self._step(torch.tensor([prompt_ids], dtype=torch.int64, device=self.device))

    @torch.inference_mode()
    def draft(self, last_token: int) -> list[int]:
        # Keep proposals on-device through the loop; sync once at the end via
        # tolist() instead of K times via item() per iteration.
        current_token = torch.tensor([[last_token]], dtype=torch.int64, device=self.device)
        draft_tokens: list[torch.Tensor] = []
        for _ in range(self.k):
            current_token = self._step(current_token).logits[0, -1].argmax().view(1, 1)
            draft_tokens.append(current_token)
        # Trailing feed: K+1-th forward adds KV for d_K so the draft cache
        # aligns with the target's verify-pass shape (see class docstring).
        self._step(current_token)
        return torch.cat(draft_tokens).view(-1).tolist()

    def rollback(self, num_reject_drafts: int) -> None:
        self.cache.crop(self.cache.get_seq_length() - num_reject_drafts)
