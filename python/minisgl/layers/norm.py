from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseOP


# --------------------------------------------------------------- helpers
def _rmsnorm_cpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # fp32 reduction — fp16 pow/mean loses precision on hidden sizes >= 1024.
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(x.dtype)


def _load_flashinfer_rmsnorm():
    try:
        from flashinfer import rmsnorm
    except ImportError as exc:
        raise RuntimeError(
            "CUDA RMSNorm requires flashinfer to be importable; install "
            "flashinfer on the host to use RMSNorm/RMSNormFused on CUDA tensors."
        ) from exc
    return rmsnorm


def _load_flashinfer_fused_add_rmsnorm():
    try:
        from flashinfer import fused_add_rmsnorm
    except ImportError as exc:
        raise RuntimeError(
            "CUDA fused-add RMSNorm requires flashinfer to be importable; "
            "install flashinfer on the host to use RMSNormFused on CUDA tensors."
        ) from exc
    return fused_add_rmsnorm


def _load_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "NPU RMSNorm requires torch_npu to be importable; install "
            "torch_npu on the host to use RMSNorm/RMSNormFused on NPU tensors."
        ) from exc
    return torch_npu


# --------------------------------------------------------------- RMSNorm
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        # No device library imports at construction time — dispatch happens on
        # the first forward, keyed off ``x.device.type``.
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        if device_type == "cuda":
            rmsnorm = _load_flashinfer_rmsnorm()
            return rmsnorm(x, self.weight, self.eps)
        if device_type == "npu":
            torch_npu = _load_torch_npu()
            # npu_rms_norm returns (y, rstd); rstd is a backward intermediate
            # unused in inference.
            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
        return _rmsnorm_cpu(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        device_type = x.device.type
        if device_type == "cuda":
            rmsnorm = _load_flashinfer_rmsnorm()
            rmsnorm(x, self.weight, self.eps, out=x)
            return None
        if device_type == "npu":
            torch_npu = _load_torch_npu()
            # NPU op has no ``out=`` — allocate then copy back into the caller's
            # buffer so the downstream rotary / attention reads see the update.
            y = torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
            x.copy_(y)
            return None
        y = _rmsnorm_cpu(x, self.weight, self.eps)
        x.copy_(y)
        return None


# ---------------------------------------------------------- RMSNormFused
class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_type = x.device.type

        if residual is None:
            # Every device returns (normalized, x) — the second slot passes the
            # original ``x`` object through as the next block's residual.
            if device_type == "cuda":
                rmsnorm = _load_flashinfer_rmsnorm()
                return rmsnorm(x, self.weight, self.eps), x
            if device_type == "npu":
                torch_npu = _load_torch_npu()
                return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0], x
            return _rmsnorm_cpu(x, self.weight, self.eps), x

        if device_type == "cuda":
            # FlashInfer semantics: fused_add_rmsnorm writes normalized(x+r)
            # back into x and (x+r) back into residual. Return the same
            # tensor objects — callers rely on this identity.
            fused_add_rmsnorm = _load_flashinfer_fused_add_rmsnorm()
            fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual

        if device_type == "npu":
            # npu_add_rms_norm returns (normalized, rstd, summed) as **fresh**
            # allocations; inputs are not mutated. The rstd slot is a backward
            # intermediate unused in inference.
            torch_npu = _load_torch_npu()
            normalized, _, summed = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.eps
            )
            return normalized, summed

        summed = x + residual
        normalized = _rmsnorm_cpu(summed, self.weight, self.eps)
        return normalized, summed
