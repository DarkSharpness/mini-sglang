from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    attn_output_gate: bool
    layer_types: list[str] | None
    full_attention_interval: int | None
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def has_linear_layers(self) -> bool:
        return self.layer_types is not None and any(t == "linear_attention" for t in self.layer_types)

    @classmethod
    def from_hf(cls, config: PretrainedConfig | dict) -> ModelConfig:
        def _get_attr(obj, attr: str):
            if isinstance(obj, dict):
                return obj.get(attr)
            return getattr(obj, attr, None)

        top = config
        text_config = _get_attr(config, "text_config")
        if text_config is not None:
            config = text_config
        if isinstance(config, dict):
            config = PretrainedConfig.from_dict(config)
        for attr in ("architectures", "rope_theta", "rope_scaling", "rope_parameters"):
            if not getattr(config, attr, None) and (top_attr := _get_attr(top, attr)):
                setattr(config, attr, top_attr)

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])
        attn_output_gate = bool(getattr(config, "attn_output_gate", False))
        full_attention_interval = getattr(config, "full_attention_interval", None)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            layer_types = list(layer_types)
        elif full_attention_interval:
            layer_types = [
                "attention" if (idx + 1) % full_attention_interval == 0 else "linear_attention"
                for idx in range(config.num_hidden_layers)
            ]

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_parameters is not None:
            rope_scaling = rope_parameters

        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None and isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0

        partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
        if partial_rotary_factor is None and isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0

        rotary_dim = int(head_dim * float(partial_rotary_factor))
        rotary_dim = max(2, min(rotary_dim, head_dim))
        if rotary_dim % 2 != 0:
            rotary_dim -= 1

        linear_conv_kernel_dim = int(getattr(config, "linear_conv_kernel_dim", 4))
        linear_key_head_dim = int(getattr(config, "linear_key_head_dim", head_dim))
        linear_value_head_dim = int(getattr(config, "linear_value_head_dim", head_dim))
        linear_num_key_heads = int(getattr(config, "linear_num_key_heads", 0))
        linear_num_value_heads = int(getattr(config, "linear_num_value_heads", 0))

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            attn_output_gate=attn_output_gate,
            layer_types=layer_types,
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
        )
