from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


@dataclass(frozen=True)
class MLAConfig:
    kv_lora_rank: int
    qk_rope_head_dim: int
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None


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
    mla_config: MLAConfig | None = None

    @property
    def is_mla(self) -> bool:
        return self.mla_config is not None

    @classmethod
    def from_hf(cls, config: LlamaConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        mla_config = None
        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        rotary_dim = head_dim
        if kv_lora_rank is not None:
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", None)
            if qk_rope_head_dim is None:
                raise ValueError(
                    "MLA model detected (kv_lora_rank present) but qk_rope_head_dim is missing."
                )
            rotary_dim = qk_rope_head_dim
            mla_config = MLAConfig(
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
                v_head_dim=getattr(config, "v_head_dim", head_dim),
            )

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
            mla_config=mla_config,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )
