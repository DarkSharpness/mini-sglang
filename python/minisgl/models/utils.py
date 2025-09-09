from __future__ import annotations

from minisgl.config.model import ModelConfig
from minisgl.layers.activation import silu_and_mul
from minisgl.layers.attention import AttentionBackend
from minisgl.layers.base import IDENTITY, BaseOP, CustomOP, ObserverOP, TakeOP
from minisgl.layers.linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
)


class GatedMLP(CustomOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        match (act_fn := getattr(config, "hidden_act", None)):
            case "silu":
                self.act_fn = silu_and_mul
            case _:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

        super().__init__(model=self.gate_up_proj + self.act_fn + self.down_proj)


class RopeAttn(CustomOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
    ):
        head_dim = config.head_dim
        attention_bias = getattr(config, "attention_bias", True)
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=attention_bias,
            qk_rms_norm_eps=config.qk_rms_norm_eps,
        )
        self.attn = AttentionBackend(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )
        super().__init__(model=self.qkv_proj + self.attn + self.o_proj)


def connect_decoder_layer(
    *,
    input_norm: BaseOP,
    self_attn: BaseOP,
    mlp: BaseOP,
    post_attn_norm: BaseOP,
    layer_id: int,
):
    from torch.cuda import nvtx

    _ = layer_id  # to avoid unused variable warning
    return (
        IDENTITY
        + ObserverOP(lambda _: nvtx.range_push(f"layer_{layer_id}"))
        + input_norm
        + ObserverOP(lambda _: nvtx.range_push(f"attn_{layer_id}"))
        + (TakeOP(0) + self_attn | TakeOP(1))
        + ObserverOP(lambda _: nvtx.range_pop())
        + post_attn_norm
        + ObserverOP(lambda _: nvtx.range_push(f"mlp_{layer_id}"))
        + (TakeOP(0) + mlp | TakeOP(1))
        + ObserverOP(lambda _: nvtx.range_pop())
        + ObserverOP(lambda _: nvtx.range_pop())
    )
