from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_test_tensor_module() -> Module:
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_test_tensor_module().test(x, y)
