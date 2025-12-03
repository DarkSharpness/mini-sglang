from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    top_k: int = 1
    ignore_eos: bool = False
