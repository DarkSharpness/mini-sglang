from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from minisgl.compilation import get_forward_context, register_split_op
from minisgl.utils import register_custom_op

from .base import BaseOP

if TYPE_CHECKING:
    from minisgl.model_executor.forward_batch_info import ForwardBatch


class RadixLinearAttention(BaseOP):
    """
    The Linear Attention Layer Implementation.
    """

    def __init__(
        self,
        layer_id: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
        # GDN KDA Shared Weights
        conv_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        activation: str = "silu",
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ):
        self.layer_id = layer_id
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_q_dim = head_q_dim
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.q_dim = num_q_heads * head_q_dim
        self.k_dim = num_k_heads * head_k_dim
        self.v_dim = num_v_heads * head_v_dim

        self.conv_weights = conv_weights
        self.bias = bias
        self.activation = activation

        self.A_log = A_log
        self.dt_bias = dt_bias

    def forward(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            # Output shape from linear attention: (1, seq_len, num_v_heads, head_v_dim)
            seq_len = mixed_qkv.shape[0]
            output = torch.empty(
                (1, seq_len, self.num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            unified_linear_attention_with_output(
                mixed_qkv,
                a,
                b,
                output,
                self.layer_id,
            )
            return output
        else:
            return forward_batch.attn_backend.forward(
                layer=self,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_linear_attention_with_output(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    """
    Custom op wrapper for linear attention computation only.
    """
    context = get_forward_context()
    if context is None:
        raise RuntimeError("No active forward context for unified_linear_attention_with_output.")

    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    if attention_layer is None:
        raise RuntimeError(f"Attention layer {layer_id} is not available in forward context.")
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    ret = forward_batch.attn_backend.forward(
        layer=attention_layer,
        forward_batch=forward_batch,
        mixed_qkv=mixed_qkv[:real_num_tokens],
        a=a[:real_num_tokens],
        b=b[:real_num_tokens],
    )

    if ret.ndim == 4:
        output[:, :real_num_tokens].copy_(ret[:, :real_num_tokens])
    elif ret.ndim == 3:
        output[:, :real_num_tokens].copy_(ret.unsqueeze(0))
    else:
        raise RuntimeError(f"Expected ret.ndim in (3, 4), got {ret.ndim}.")
    return
