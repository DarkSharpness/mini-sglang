from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


def create_attention_backend(
    config: ModelConfig, base_kvcache: BaseKVCache, backend: str
) -> BaseAttnBackend:
    match backend:
        case "fa3":
            from .fa3 import FlashAttentionBackend

            return FlashAttentionBackend(config, base_kvcache)
        case "fi":
            from .fi import FlashInferBackend

            return FlashInferBackend(config, base_kvcache)
        case _:
            raise ValueError(f"Unsupported attention backend: {backend}")


__all__ = ["BaseAttnMetadata", "BaseAttnBackend", "create_attention_backend"]
