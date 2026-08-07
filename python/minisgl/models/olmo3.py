from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    DistributedRMSNorm,
    LinearOProj,
    LinearQKVMerged,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import RotaryConfig
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


class Olmo3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp_size = get_tp_info().size
        if config.num_kv_heads < tp_size:
            raise NotImplementedError(
                "OLMo3 TP with replicated KV heads requires replica-aware K RMSNorm"
            )
        assert config.layer_types is not None
        layer_type = config.layer_types[layer_id]
        rotary_config = config.rotary_config
        if layer_type == "sliding_attention":
            rotary_config = RotaryConfig(
                head_dim=rotary_config.head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
                scaling=None,
            )
        else:
            assert layer_type == "full_attention"

        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        q_projection_size = config.num_qo_heads * config.head_dim
        k_projection_size = config.num_kv_heads * config.head_dim
        norm_cls = DistributedRMSNorm if tp_size > 1 else RMSNorm
        self.q_norm = norm_cls(q_projection_size, config.rms_norm_eps)
        self.k_norm = norm_cls(k_projection_size, config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            rotary_config=rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            qk_norm_mode="projection",
        )
        self.o_proj = LinearOProj(
            input_size=config.num_qo_heads * config.head_dim,
            output_size=config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        output = self.attn.forward(qkv)
        return self.o_proj.forward(output)


class Olmo3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Olmo3Attention(config, layer_id)
        self.mlp = GatedMLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn.forward(x)
        x = self.post_attention_layernorm.forward(x)
        x = residual + x

        residual = x
        x = self.mlp.forward(x)
        x = self.post_feedforward_layernorm.forward(x)
        return residual + x


class Olmo3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Olmo3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Olmo3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Olmo3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Olmo3ForCausalLM"]
