from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    KVCacheLayout,
    KVCacheType,
    SizeInfo,
)


def create_kvcache(
    num_layers: int,
    num_kv_heads: int,
    num_pages: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
    cache_type: KVCacheType = KVCacheType.MHA,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache

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
    raise ValueError(f"Unsupported KVCacheType: {cache_type}")


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    match type:
        case "radix":
            from .radix_manager import RadixCacheManager

            return RadixCacheManager(device=device)
        case "naive":
            from .naive_manager import NaiveCacheManager

            return NaiveCacheManager(device=device)
    raise ValueError(f"Unsupported cache manager type: {type}")


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "BaseKVCache",
    "KVCacheLayout",
    "KVCacheType",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
]
