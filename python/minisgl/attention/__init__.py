from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AttnArgs, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCache


def create_attention_backend(base_kvcache: BaseKVCache, backend: str) -> BaseAttnBackend:
    from .fa3 import FlashAttentionBackend

    match backend:
        case "fa3":
            return FlashAttentionBackend(base_kvcache)
        case _:
            raise ValueError(f"Unsupported attention backend: {backend}")


__all__ = ["BaseAttnMetadata", "BaseAttnBackend", "AttnArgs", "create_attention_backend"]
