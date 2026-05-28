from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    Gemma3RMSNorm,
    LinearOProj,
    LinearQKVMerged,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import RotaryConfig
from .utils import GatedMLP as Gemma3MLP

if TYPE_CHECKING:
    from .config import ModelConfig


class Gemma3Attn(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        is_sliding = (
            config.layer_types[layer_id] == "sliding_attention" if config.layer_types else False
        )
        rotary_dim = int(config.partial_rotary_factor * head_dim)
        rope_theta = (
            config.local_rope_theta if is_sliding else config.global_rope_theta
        ) or config.rotary_config.base
        softmax_scale = (
            config.query_pre_attn_scalar**-0.5 if config.query_pre_attn_scalar is not None else None
        )
        sliding_window_size = config.sliding_window if is_sliding else None
        assert not is_sliding or sliding_window_size is not None

        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=config.attention_bias,
        )
        self.q_norm = Gemma3RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.rotary_config.max_position,
                base=rope_theta,
                scaling=config.rotary_config.scaling,
            ),
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            sliding_window_size=sliding_window_size,
            softmax_scale=softmax_scale,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=config.attention_bias,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


class Gemma3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Gemma3Attn(config, layer_id)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)
        x = self.post_attention_layernorm.forward(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm.forward(x)
        x = self.mlp.forward(x)
        x = self.post_feedforward_layernorm.forward(x)
        return residual + x


class Gemma3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Gemma3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = Gemma3RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.embed_scale = math.sqrt(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids) * self.embed_scale
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Gemma3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Gemma3ForCausalLM"]
