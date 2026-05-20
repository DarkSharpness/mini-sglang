from typing import Tuple

import torch

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


class GemmaRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import gemma_rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gemma_rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.gemma_rmsnorm(x, self.weight, self.eps, out=x)


class GemmaRMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import gemma_fused_add_rmsnorm, gemma_rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm
        self.gemma_fused_add_rmsnorm = gemma_fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.gemma_rmsnorm(x, self.weight, self.eps), x
        self.gemma_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
