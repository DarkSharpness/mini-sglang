from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.compilation import get_forward_context
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    GemmaRMSNormFused,
    LinearColParallelMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RadixLinearAttention,
    VocabParallelEmbedding,
)
from minisgl.kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3_5MLP
from .utils import RopeAttn as Qwen3_5Attn

if TYPE_CHECKING:
    from .config import ModelConfig


def _get_layer_type(config: "ModelConfig", layer_id: int) -> str:
    if not config.layer_types:
        return "attention"
    layer_type = config.layer_types[layer_id]
    if layer_type == "full_attention":
        return "attention"
    return layer_type


class _DepthwiseConv1D(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Qwen3_5GatedDeltaNet(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        if config.linear_num_key_heads <= 0 or config.linear_num_value_heads <= 0:
            raise ValueError("Invalid qwen3.5 linear attention config.")

        tp_size = get_tp_info().size
        self.layer_id = layer_id
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_q_heads = div_even(config.linear_num_key_heads, tp_size, allow_replicate=True)
        self.num_v_heads = div_even(config.linear_num_value_heads, tp_size, allow_replicate=True)
        if self.num_v_heads % self.num_q_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_q_heads for GDN.")

        key_dim = config.linear_num_key_heads * self.head_k_dim
        value_dim = config.linear_num_value_heads * self.head_v_dim
        self.hidden_act = config.hidden_act

        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=config.hidden_size,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=config.hidden_size,
            num_v_heads=config.linear_num_value_heads,
        )

        self.local_qk_dim = self.num_q_heads * self.head_k_dim
        self.local_v_dim = self.num_v_heads * self.head_v_dim
        self.local_conv_dim = 2 * self.local_qk_dim + self.local_v_dim

        self.conv1d = _DepthwiseConv1D(self.local_conv_dim, config.linear_conv_kernel_dim)
        self.A_log = torch.empty(self.num_v_heads)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_q_heads,
            num_k_heads=self.num_q_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            # Keep these as non-Tensor at init time to avoid state_dict loading
            # expecting duplicated keys under linear_attn.attn.*.
            conv_weights=None,
            bias=None,
            activation=self.hidden_act,
            A_log=None,
            dt_bias=None,
        )

        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(
            value_dim,
            config.hidden_size,
            has_bias=False,
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
    ) -> LinearColParallelMerged:
        return LinearColParallelMerged(
            hidden_size,
            [key_dim, key_dim, value_dim, value_dim],
            has_bias=False,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
    ) -> LinearColParallelMerged:
        return LinearColParallelMerged(
            hidden_size,
            [num_v_heads, num_v_heads],
            has_bias=False,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value, z = mixed_qkvz.split(
            [self.local_qk_dim, self.local_qk_dim, self.local_v_dim, self.local_v_dim], dim=-1
        )
        b, a = mixed_ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        value = value.view(value.shape[0], self.num_v_heads, self.head_v_dim)
        z = z.view(z.shape[0], self.num_v_heads, self.head_v_dim)
        return query, key, value, z, b, a

    def _forward_input_proj(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_states_qkvz = self.in_proj_qkvz.forward(x)
        projected_states_ba = self.in_proj_ba.forward(x)
        return projected_states_qkvz, projected_states_ba

    def _fused_input_reorder(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not projected_states_qkvz.is_cuda:
            return None
        if self.num_v_heads % self.num_q_heads != 0:
            return None
        if (self.num_v_heads // self.num_q_heads) not in (1, 2, 4):
            return None
        try:
            return fused_qkvzba_split_reshape_cat_contiguous(
                projected_states_qkvz,
                projected_states_ba,
                self.num_q_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
        except Exception:
            return None

    @nvtx_annotate("GatedDeltaNet")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_ctx = get_forward_context()
        forward_batch = forward_ctx.forward_batch if forward_ctx is not None else None

        projected_states_qkvz, projected_states_ba = self._forward_input_proj(x)
        fused_out = self._fused_input_reorder(projected_states_qkvz, projected_states_ba)
        if fused_out is None:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            mixed_qkv = torch.cat((query, key, value.reshape(value.shape[0], -1)), dim=-1)
        else:
            mixed_qkv, z, b, a = fused_out
        b = b.contiguous()
        a = a.contiguous()

        # Keep RadixLinearAttention references aligned after state_dict loading.
        self.attn.conv_weights = self.conv1d.weight
        self.attn.A_log = self.A_log
        self.attn.dt_bias = self.dt_bias
        core = self.attn.forward(
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        ).to(dtype=z.dtype)

        core = core.reshape(-1, core.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core = self.norm.forward(core)
        core = core * F.silu(z)
        core = core.reshape(x.shape[0], -1)
        return self.out_proj.forward(core)


class _Qwen3_5BaseDecoderLayer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self._run_attention(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5LinearDecoderLayer(_Qwen3_5BaseDecoderLayer):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_id)
        super().__init__(config, layer_id)

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_attn.forward(x)


class Qwen3_5AttentionDecoderLayer(_Qwen3_5BaseDecoderLayer):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self.self_attn = Qwen3_5Attn(
            config,
            layer_id,
            has_qk_norm=True,
            has_attn_output_gate=config.attn_output_gate,
            use_gemma_norm=True,
        )
        super().__init__(config, layer_id)

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.forward(x)


_DECODER_LAYER_REGISTRY = {
    "linear_attention": Qwen3_5LinearDecoderLayer,
    "attention": Qwen3_5AttentionDecoderLayer,
}


class Qwen3_5Model(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        layers: list[BaseOP] = []
        for layer_id in range(config.num_layers):
            layer_type = _get_layer_type(config, layer_id)
            layer_cls = _DECODER_LAYER_REGISTRY.get(layer_type)
            if layer_cls is None:
                raise ValueError(f"Unsupported qwen3.5 layer type: {layer_type}")
            layers.append(layer_cls(config, layer_id))
        self.layers = OPList(layers)
        self.norm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = Qwen3_5Model(config)
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


__all__ = ["Qwen3_5GatedDeltaNet", "Qwen3_5ForCausalLM"]
