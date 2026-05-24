from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, DynamicCache


class StandaloneDraft:
    """
    Autoregressive draft backed by a smaller standalone model from the target's family.
    Advances its cache by K+1 positions per round (K forwards to generate the
    candidates plus one trailing feed for the K-th token's KV) so it stays aligned
    with the target's K+1-wide verify pass and rollback is a plain truncation.
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
        # aligns with the target's verify-pass shape.
        self._step(current_token)
        return torch.cat(draft_tokens).view(-1).tolist()

    def rollback(self, num_reject_drafts: int) -> None:
        self.cache.crop(self.cache.get_seq_length() - num_reject_drafts)
