from __future__ import annotations

from typing import TYPE_CHECKING, override

from minisgl.config.context import get_global_ctx
from minisgl.layers.base import CustomOP, ListOP, TakeOP
from minisgl.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from minisgl.layers.norm import RMSNormFused

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn
from .utils import connect_decoder_layer

if TYPE_CHECKING:
    import torch

    from .config import ModelConfig


class LlamaDecoderLayer(CustomOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = LlamaAttn(config, layer_id, has_bias=False)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        super().__init__(
            model=connect_decoder_layer(
                input_norm=self.input_layernorm,
                self_attn=self.self_attn,
                post_attn_norm=self.post_attention_layernorm,
                mlp=self.mlp,
                layer_id=layer_id,
            )
        )


class LlamaModel(CustomOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = ListOP(
            LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        super().__init__(model=self.embed_tokens + self.layers + self.norm + TakeOP(0))


class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        super().__init__()

    @override
    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        output: torch.Tensor = self.model.forward(ctx.batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
