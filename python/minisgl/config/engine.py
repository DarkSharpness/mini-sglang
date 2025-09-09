from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List

import torch
from minisgl.config.model import ModelConfig
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int
    cuda_graph_bs: List[int] = field(default_factory=list)
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self) -> ModelConfig:
        model_config = cached_load_hf_config(self.model_path)
        return ModelConfig.from_hf(model_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.hf_config.rotary_config.max_position
