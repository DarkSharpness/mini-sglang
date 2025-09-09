from __future__ import annotations

from typing import List, final

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP
from .norm import RMSNorm


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


@final
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


@final
class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        qk_rms_norm_eps: float | None = None,  # whether to apply RMSNorm to q and k
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

        # maybe we have q/k norm
        self.q_norm = RMSNorm(size=head_dim, eps=qk_rms_norm_eps) if qk_rms_norm_eps else None
        self.k_norm = RMSNorm(size=head_dim, eps=qk_rms_norm_eps) if qk_rms_norm_eps else None

        # store some additional information
        self._head_dim = head_dim
        self._qo_attn_dim = GQA_ratio * local_num_kv * head_dim
        self._kv_attn_dim = local_num_kv * head_dim

    def _apply_qk_norm_inplace(self, q: torch.Tensor, k: torch.Tensor):
        if self.q_norm is not None:
            assert self.k_norm is not None
            head_dim = self._head_dim
            q = q.view(-1, head_dim)
            k = k.view(-1, head_dim)
            self.q_norm.forward_(q)
            self.k_norm.forward_(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = super().forward(x)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            q, k, _ = qkv.split([self._qo_attn_dim, self._kv_attn_dim, self._kv_attn_dim], dim=-1)
            self._apply_qk_norm_inplace(q, k)
        return qkv


@final
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


@final
class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
