import torch

from .base import BaseKVCache, KVCacheLayout, KVCacheType


def create_kvcache(
    num_layers: int,
    num_kv_heads: int,
    num_pages: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.PageFirst,
    cache_type: KVCacheType = KVCacheType.MHA,
) -> BaseKVCache:
    from .mha import MHAKVCache

    match cache_type:
        case KVCacheType.MHA:
            return MHAKVCache(
                num_kv_heads=num_kv_heads,
                num_pages=num_pages,
                kv_layout=cache_layout,
                num_layers=num_layers,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
        case _:
            raise ValueError(f"Unsupported KVCacheType: {cache_type}")


__all__ = ["create_kvcache", "BaseKVCache", "KVCacheLayout", "KVCacheType"]
