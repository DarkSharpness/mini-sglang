from __future__ import annotations

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .weight import load_hf_weight


def create_model(config: ModelConfig) -> BaseLLMModel:
    model_path = config.model_path
    model_config = config.model_config
    model_name = model_path.lower()
    if "llama" in model_name:
        from .llama import LlamaForCausalLM

        return LlamaForCausalLM(model_config)
    elif "qwen3" in model_name and "30b" in model_name:
        from .qwen3_moe import Qwen3MoeForCausalLM

        moe_backend = config.moe_backend
        return Qwen3MoeForCausalLM(model_config, moe_backend)
    elif "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_path}")


__all__ = ["BaseLLMModel", "load_hf_weight", "create_model", "ModelConfig", "RotaryConfig"]
