"""Weight quantization utilities for mini-sglang."""

from __future__ import annotations

import torch

from .config import QuantizationConfig, QuantizationScheme


def quantize_weight_int8_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to INT8 using per-channel quantization.

    Args:
        weight: FP32/FP16/BF16 weight tensor of shape [out_features, in_features]

    Returns:
        Tuple of (quantized_weight, scale, zero_point)
        - quantized_weight: INT8 tensor
        - scale: Per-channel scale factors (one per output channel)
        - zero_point: Per-channel zero points
    """
    # Compute per-channel (per output feature) min/max
    out_features = weight.shape[0]

    # Flatten along input dimension to get per-channel stats
    weight_float = weight.float()  # Convert to FP32 for precision
    min_vals = weight_float.view(out_features, -1).min(dim=1)[0]
    max_vals = weight_float.view(out_features, -1).max(dim=1)[0]

    # Compute scale and zero_point for symmetric quantization
    # INT8 range: [-128, 127]
    scale = (max_vals - min_vals) / 255.0
    scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero

    zero_point = torch.round(-min_vals / scale).to(torch.int8)

    # Quantize: Q = round(W / scale) + zero_point
    quantized = torch.clamp(
        torch.round(weight_float / scale.view(-1, 1)) + zero_point.view(-1, 1),
        -128,
        127,
    ).to(torch.int8)

    return quantized, scale, zero_point


def quantize_weight_int8_per_tensor(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to INT8 using per-tensor quantization.

    Args:
        weight: FP32/FP16/BF16 weight tensor

    Returns:
        Tuple of (quantized_weight, scale, zero_point)
    """
    weight_float = weight.float()
    min_val = weight_float.min()
    max_val = weight_float.max()

    scale = (max_val - min_val) / 255.0
    scale = torch.clamp(scale, min=1e-8)

    zero_point = torch.round(-min_val / scale).to(torch.int8)

    quantized = torch.clamp(
        torch.round(weight_float / scale) + zero_point,
        -128,
        127,
    ).to(torch.int8)

    return quantized, scale.view(1), zero_point.view(1)


def quantize_weight(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Quantize a weight tensor according to the configuration.

    Args:
        weight: Weight tensor to quantize
        config: Quantization configuration

    Returns:
        Tuple of (quantized_weight, scale, zero_point) or None if quantization disabled
    """
    if not config.enabled:
        return None

    if config.scheme == QuantizationScheme.INT8_PER_CHANNEL:
        return quantize_weight_int8_per_channel(weight)
    elif config.scheme == QuantizationScheme.INT8_PER_TENSOR:
        return quantize_weight_int8_per_tensor(weight)
    else:
        return None


def dequantize_weight(
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an INT8 weight tensor back to FP32.

    Args:
        quantized_weight: INT8 weight tensor
        scale: Scale factors
        zero_point: Zero points

    Returns:
        Dequantized FP32 tensor
    """
    # W = (Q - zero_point) * scale
    weight_float = (quantized_weight.float() - zero_point.float()) * scale.float()
    return weight_float
