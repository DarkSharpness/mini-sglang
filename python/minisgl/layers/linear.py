from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.quantization import dequantize_weight
from minisgl.utils import divide_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism.

    Supports both FP and INT8 quantized weights with on-the-fly dequantization.
    """

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

        # Quantization metadata (None if not quantized)
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None
        self.is_quantized = False

    def _get_weight(self) -> torch.Tensor:
        """Get weight tensor, dequantizing if necessary."""
        if self.is_quantized:
            assert self.scale is not None and self.zero_point is not None
            return dequantize_weight(self.weight, self.scale, self.zero_point)
        return self.weight

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Override to handle quantized weights and metadata."""
        from .base import _concat_prefix

        # Check if this layer has quantized weights
        weight_key = _concat_prefix(prefix, "weight")
        scale_key = f"{weight_key}.scale"
        zero_point_key = f"{weight_key}.zero_point"

        has_scale = scale_key in state_dict
        has_zero_point = zero_point_key in state_dict

        if has_scale and has_zero_point:
            # Quantized weights
            weight = state_dict.pop(weight_key)
            scale = state_dict.pop(scale_key)
            zero_point = state_dict.pop(zero_point_key)

            assert weight.dtype == torch.int8, f"Expected int8 weight, got {weight.dtype}"
            assert weight.shape == self.weight.shape, f"Shape mismatch: {weight.shape} vs {self.weight.shape}"

            self.weight = weight
            self.scale = scale
            self.zero_point = zero_point
            self.is_quantized = True

            # Handle bias if present
            bias_key = _concat_prefix(prefix, "bias")
            if bias_key in state_dict and self.bias is not None:
                self.bias = state_dict.pop(bias_key)
        else:
            # Non-quantized weights - use base class behavior
            super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._get_weight()
        return F.linear(x, weight, self.bias)


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


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


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
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


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
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
