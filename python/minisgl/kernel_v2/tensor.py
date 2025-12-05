from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_test_tensor_module() -> Module:
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_test_tensor_module().test(x, y)


def make_2d_indices(
    table_2d: torch.Tensor,
    ranges: List[Tuple[int, int, int]],
    *,
    load_table: bool,
    store_value: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch

    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    stride = table_2d.stride(0)
    buffer_size = sum(end - begin for _, begin, end in ranges)
    host_buffer = torch.empty(buffer_size, dtype=torch.int32, pin_memory=True)
    global_offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        offset = stride * entry
        torch.arange(
            begin + offset,
            end + offset,
            dtype=torch.int32,
            out=host_buffer[global_offset : global_offset + length],
        )
        global_offset += length

    indices = host_buffer.to(table_2d.device, non_blocking=True)
    if store_value is not None:
        table_2d.view(-1)[indices] = store_value
    return table_2d.view(-1)[indices] if load_table else indices
