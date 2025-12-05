from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    top_k: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    max_tokens: int = 1024
