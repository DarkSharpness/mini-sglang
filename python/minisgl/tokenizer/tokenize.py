from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager: 
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        if not msgs:
            return []

        # Separate plain text and chat template messages while preserving order
        plain_indices: List[int] = []
        plain_texts: List[str] = []
        chat_indices: List[int] = []
        chat_convs: List[List[dict]] = []

        for i, msg in enumerate(msgs):
            if isinstance(msg.text, list):
                chat_indices.append(i)
                chat_convs.append(msg.text)
            else:
                plain_indices.append(i)
                plain_texts.append(msg.text)

        results: List[torch.Tensor | None] = [None] * len(msgs)

        # Batch encode plain texts
        if plain_texts:
            encoded = self.tokenizer(plain_texts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                # Remove padding tokens
                length = mask.sum().item()
                results[plain_indices[i]] = ids[:length].to(torch.int32)

        # Batch encode chat templates
        if chat_convs:
            prompts = self.tokenizer.apply_chat_template(
                chat_convs,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                # Remove padding tokens
                length = mask.sum().item()
                results[chat_indices[i]] = ids[:length].to(torch.int32)

        return results  # type: ignore
