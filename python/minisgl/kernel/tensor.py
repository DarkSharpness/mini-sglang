from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_validate_tensor_ffi_module() -> Module:
    return load_aot("validate_tensor_ffi", cpp_files=["tensor.cpp"])


def validate_tensor_ffi(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_validate_tensor_ffi_module().test(x, y)
