from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.config.context import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.models import RotaryConfig


class AttentionLayer(BaseOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self._layer_id = layer_id
        self._head_dim = head_dim
        tp_size = get_tp_info().size
        self._num_qo_heads = divide_even(num_qo_heads, tp_size)
        self._num_kv_heads = divide_even(num_kv_heads, tp_size)
        self._qo_attn_dim = self._num_qo_heads * head_dim
        self._kv_attn_dim = self._num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        q, k, v = qkv.split([self._qo_attn_dim, self._kv_attn_dim, self._kv_attn_dim], dim=-1)
        if self.rotary:
            q, k = self.rotary.forward(metadata.get_positions(), q, k)
        q = q.view(-1, self._num_qo_heads, self._head_dim)
        o = ctx.attn_backend.forward(q, k, v, self._layer_id, ctx.batch)
        return o.view(-1, self._qo_attn_dim)
