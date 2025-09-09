from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

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
    kv_buffer: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
    module.store_cache(k_cache, v_cache, out_loc, k, v, kv_buffer)
