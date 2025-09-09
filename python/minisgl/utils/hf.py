from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import LlamaConfig


@lru_cache()
def _load_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> LlamaConfig:
    # deep copy the config to avoid modifying the original config
    config = _load_config(model_path)
    return type(config)(**config.to_dict())
