from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    import torch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


def create_attention_backend(
    config: ModelConfig,
    base_kvcache: BaseKVCache,
    backend: str,
    page_table: torch.Tensor,
) -> BaseAttnBackend:
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        prefill_backend, decode_backend = backend.split(",", 1)
        if prefill_backend != decode_backend:
            prefill_backend = create_attention_backend(
                config, base_kvcache, prefill_backend, page_table
            )
            decode_backend = create_attention_backend(
                config, base_kvcache, decode_backend, page_table
            )
            return HybridBackend(prefill_backend, decode_backend)
        backend = prefill_backend

    match backend:
        case "fa3":
            from .fa3 import FlashAttentionBackend

            return FlashAttentionBackend(config, base_kvcache, page_table)
        case "fi":
            from .fi import FlashInferBackend

            return FlashInferBackend(config, base_kvcache, page_table)
        case _:
            raise ValueError(f"Unsupported attention backend: {backend}")


__all__ = ["BaseAttnMetadata", "BaseAttnBackend", "create_attention_backend"]
