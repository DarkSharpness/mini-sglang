from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(msg.text, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
