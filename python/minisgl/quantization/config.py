"""Quantization configuration for mini-sglang."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class QuantizationScheme(str, Enum):
    """Supported quantization schemes."""

    INT8_PER_CHANNEL = "int8_per_channel"
    INT8_PER_TENSOR = "int8_per_tensor"
    NONE = "none"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        scheme: Quantization scheme to use
        dtype: Target dtype for quantized weights (int8)
        enabled: Whether quantization is enabled
    """

    scheme: QuantizationScheme = QuantizationScheme.NONE
    dtype: torch.dtype = torch.int8
    enabled: bool = False

    @classmethod
    def from_scheme(cls, scheme: str | QuantizationScheme | None) -> QuantizationConfig:
        """Create config from scheme name.

        Args:
            scheme: Quantization scheme name or None to disable

        Returns:
            QuantizationConfig instance
        """
        if scheme is None or scheme == "none":
            return cls(scheme=QuantizationScheme.NONE, enabled=False)

        if isinstance(scheme, str):
            scheme = QuantizationScheme(scheme)

        return cls(scheme=scheme, enabled=True)

    def __repr__(self) -> str:
        if not self.enabled:
            return "QuantizationConfig(disabled)"
        return f"QuantizationConfig(scheme={self.scheme.value}, dtype={self.dtype})"
