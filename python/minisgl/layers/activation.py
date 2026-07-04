from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if x.shape[-1] % 2 != 0:
        raise ValueError("silu_and_mul requires an even last dimension")

    expected_shape = (*x.shape[:-1], x.shape[-1] // 2)
    if out is not None:
        if tuple(out.shape) != tuple(expected_shape):
            raise ValueError(
                f"silu_and_mul: out.shape {tuple(out.shape)} does not match "
                f"expected {tuple(expected_shape)}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"silu_and_mul: out.dtype {out.dtype} does not match x.dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError(
                f"silu_and_mul: out.device {out.device} does not match x.device {x.device}"
            )

    device_type = x.device.type

    if device_type == "cuda":
        try:
            from flashinfer import silu_and_mul as flashinfer_silu_and_mul
        except ImportError as exc:
            raise RuntimeError(
                "CUDA silu_and_mul requires flashinfer to be importable"
            ) from exc

        return flashinfer_silu_and_mul(x, out=out)

    if device_type == "npu":
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError(
                "NPU silu_and_mul requires torch_npu to be importable"
            ) from exc

        result = torch_npu.npu_swiglu(x, dim=-1)

        if out is not None:
            out.copy_(result)
            return out

        return result

    gate, up = x.float().chunk(2, dim=-1)
    result = (F.silu(gate) * up).to(x.dtype)

    if out is not None:
        out.copy_(result)
        return out

    return result


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
