from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from minisgl.utils import Registry, init_logger, is_sm90_supported

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)

logger = init_logger(__name__)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    kv_dtype: torch.dtype | None = None,
    attention_backend: str | None = None,
) -> BaseKVCachePool:
    if kv_dtype is None or kv_dtype == dtype:
        from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

        cls: type[BaseKVCachePool] = MHAKVCache
    elif kv_dtype == torch.float8_e4m3fn:
        # FA fp8 KV requires sm_90+ (FA3 on Hopper, FA4 on Blackwell). Refuse
        # the combination on pre-Hopper hardware -- the kernel accepts fp8
        # tensors at the Python boundary but the underlying SASS path is
        # missing, so the failure mode is silent corruption.
        if (
            attention_backend
            and "fa" in attention_backend.split(",")
            and not is_sm90_supported()
        ):
            major, minor = torch.cuda.get_device_capability(device)
            raise ValueError(
                f"FP8 KV cache with the FlashAttention backend requires sm_90 "
                f"(Hopper) or sm_100 (Blackwell). Detected sm_{major}{minor}. "
                f"Use --attention-backend fi (FlashInfer) instead, or run on a "
                f"supported GPU."
            )
        logger.warning_rank0(
            "FP8 KV cache enabled with scale=1.0. K/V are clamped to +/-448 before "
            "cast; outliers beyond +/-448 saturate. Expect quality regression on "
            "long-context or outlier-heavy workloads. Plumbing-only; calibrated "
            "k_scale/v_scale in checkpoints are ignored in this version."
        )
        from .quantized_mha_pool import QuantizedMHAKVCache

        cls = QuantizedMHAKVCache
    else:
        raise ValueError(
            f"Unsupported kv_dtype {kv_dtype}; only torch.float8_e4m3fn is supported."
        )

    return cls(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
