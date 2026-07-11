from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        prompts: List[str] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            prompts.append(prompt)

        if not prompts:
            return []

        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        return [torch.tensor(input_ids, dtype=torch.int32) for input_ids in encoded]
