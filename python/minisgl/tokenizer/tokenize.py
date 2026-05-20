from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        if len(msgs) == 0:
            return []

        # Step 1: Prepare all prompts (handle both chat templates and plain text)
        prompts: List[str] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                # Chat format - apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                # Plain text
                prompt = msg.text
            prompts.append(prompt)

        # Step 2: Batch tokenization (no padding for efficiency)
        # Returns BatchEncoding with input_ids as List[List[int]]
        batch_result = self.tokenizer(
            prompts,
            padding=False,
            add_special_tokens=True,
        )

        # Step 3: Convert each result to individual tensor
        results: List[torch.Tensor] = [
            torch.tensor(input_ids, dtype=torch.int32)
            for input_ids in batch_result["input_ids"]
        ]

        return results
