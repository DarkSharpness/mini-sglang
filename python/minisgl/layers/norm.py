from typing import Tuple

import torch
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class DistributedRMSNorm(BaseOP):
    """RMSNorm over a projection sharded across tensor-parallel ranks."""

    def __init__(self, full_size: int, eps: float) -> None:
        tp_info = get_tp_info()
        assert tp_info.size > 1
        self.eps = eps
        self.full_size = full_size
        self.weight = torch.empty(div_even(full_size, tp_info.size))
        self._comm = DistributedCommunicator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.clone()
        self.forward_inplace(output)
        return output

    def forward_inplace(self, x: torch.Tensor) -> None:
        assert x.shape[-1] == self.weight.shape[0]
        square_sum = x.float().square().sum(dim=-1, keepdim=True)
        square_sum = self._comm.all_reduce(square_sum)
        inv_rms = torch.rsqrt(square_sum / self.full_size + self.eps)
        x.copy_((x.float() * inv_rms * self.weight.float()).to(x.dtype))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
