from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    from flashinfer import silu_and_mul

    return silu_and_mul(x)


__all__ = ["silu_and_mul"]
