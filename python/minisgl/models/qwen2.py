from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx, Batch
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen2MLP
from .utils import RopeAttn as Qwen2Attn

if TYPE_CHECKING:
    from .config import ModelConfig

DEFAULT_BLOCK_SIZE = 4

class Qwen2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen2Attn(config, layer_id, has_qk_norm=False, has_attn_bias=True)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual
    
    @nvtx_annotate("Layer_{}_ProjectOnly", layer_id_field="_layer_id")
    def forward_project_only(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # project-only forward pass that bypasses self.mlp, self.o_proj, and attention mixing

        # normalise the hidden state to compute accurate K/V
        norm_x, _ = self.input_layernorm.forward(x, residual)
        
        # compute QKV and store in cache
        self.self_attn.forward_project_only(norm_x)
        
        # return the original x and residual
        return x, residual


class Qwen2Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen2DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # model blocks
        self.BLOCK_SIZE = DEFAULT_BLOCK_SIZE
        self.blocks = [
            self.layers.op_list[i : i + self.BLOCK_SIZE] 
            for i in range(0, config.num_layers, self.BLOCK_SIZE)
        ]

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return x, residual
    
    def forward_block_compute(
        self, block_idx: int, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.blocks[block_idx]:
            x, residual = layer.forward(x, residual)
        return x, residual

    def forward_block_project_only(
        self, block_idx: int, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.blocks[block_idx]:
            x, residual = layer.forward_project_only(x, residual)
        return x, residual


class Qwen2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # standard forward function, used for prefill with dynamic shapes
        output, _ = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits
    
    def forward_single_block(self, block_idx: int, is_project: bool, batch: Batch) -> torch.Tensor | None:
        # executes one block, gathering and scattering from the VRAM ledger
        ctx = get_global_ctx()
        table_indices = batch.table_indices

        # gather state from VRAM ledger, or embed tokens if first block
        if block_idx == 0:
            x = self.model.embed_tokens.forward(batch.input_ids)
            residual = None
        else:
            x = ctx.engine.global_hidden_states[table_indices]
            residual = ctx.engine.global_residuals[table_indices]

        # compute block
        if is_project:
            x, residual = self.model.forward_block_project_only(block_idx, x, residual)
        else:
            x, residual = self.model.forward_block_compute(block_idx, x, residual)

        # scatter state back to ledger or return logits if fina
        if block_idx == len(self.model.blocks) - 1:
            # final block computes the LM head output - no matter if we are skipping/is_project or not
            output = self.model.norm.forward(x, residual)[0]
            logits = self.lm_head.forward(output)
            return logits
        else:
            # intermediate blocks write their output back to the global VRAM ledger
            ctx.engine.global_hidden_states[table_indices] = x
            if residual is not None:
                ctx.engine.global_residuals[table_indices] = residual
            return None


__all__ = ["Qwen2ForCausalLM"]
