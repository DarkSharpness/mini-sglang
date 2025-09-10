from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Tuple

from .utils import load_kernel_module

if TYPE_CHECKING:
    import torch


def _load_kvcache_module() -> Any:
    return load_kernel_module("kvcache.cu", "kvcache_kernel")


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """
    Store key-value cache in the given tensors.
    """
    module = _load_kvcache_module()
    max_tokens = k_cache.size(0)
    num_tokens = out_loc.size(0)
    k_cache = k_cache.view(max_tokens, -1)
    v_cache = v_cache.view(max_tokens, -1)
    k = k.view(num_tokens, -1)
    v = v.view(num_tokens, -1)
    module.store_cache(k_cache, v_cache, out_loc, k, v)


@lru_cache(maxsize=1)
def _lru_get_pos(pos_list: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    import torch

    """
    This position can be hopefully reused, especially for patterns that
    a load often happens right after a store.
    """
    pos_tensor = torch.tensor(pos_list, device="cpu", dtype=torch.int32, pin_memory=True)
    return pos_tensor.to(device, non_blocking=True)


def store_decode_indices(
    page_table: torch.Tensor,
    indices: torch.Tensor,
    pos: List[Tuple[int, int]],
) -> None:
    """
    Write decode indices to the given page_table.
    """
    page_table_len = page_table.size(1)
    flattened_table = page_table.view(-1)
    pos_list = tuple(p[0] * page_table_len + p[1] for p in pos)
    pos_tensor = _lru_get_pos(tuple(pos_list), page_table.device)
    flattened_table[pos_tensor] = indices


def load_decode_indices(
    page_table: torch.Tensor,
    pos: List[Tuple[int, int]],
) -> torch.Tensor:
    page_table_len = page_table.size(1)
    flattened_table = page_table.view(-1)
    pos_list = tuple(p[0] * page_table_len + p[1] for p in pos)
    pos_tensor = _lru_get_pos(tuple(pos_list), page_table.device)
    return flattened_table[pos_tensor]
