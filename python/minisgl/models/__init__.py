from __future__ import annotations

from minisgl.config.model import ModelConfig

from .base import BaseLLMModel
from .weight import load_hf_weight


def create_model(model_path: str, model_config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "llama" in model_name:
        from .llama import LlamaForCausalLM

        return LlamaForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_path}")


__all__ = ["BaseLLMModel", "load_hf_weight", "create_model"]
