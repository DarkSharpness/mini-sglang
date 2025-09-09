import torch
from sgl_kernel.elementwise import fused_add_rmsnorm, rmsnorm

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(RMSNorm):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__(size, eps)

    def _forward_fused(self, x: torch.Tensor, residual: torch.Tensor):
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return super().forward(x), x
        return self._forward_fused(x, residual)
