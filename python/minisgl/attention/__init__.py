from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from minisgl.utils import Registry, init_logger, is_sm90_supported, is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


class BackendCreator(Protocol):
    def __call__(
        self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor
    ) -> BaseAttnBackend: ...


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


def _resolve_auto_backend(config: ModelConfig) -> str:
    if is_sm100_supported():  # blackwell
        return "fi"
    elif is_sm90_supported():  # hopper
        return "fa3,fi"
    else:  # pre-hopper
        return "fi"


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
    from .fi import FlashInferBackend

    return FlashInferBackend(config, kvcache, page_table)


@SUPPORTED_ATTENTION_BACKENDS.register("fa3")
def create_fa3_backend(config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
    from .fa3 import FlashAttentionBackend

    return FlashAttentionBackend(config, kvcache, page_table)


def create_attention_backend(
    config: ModelConfig,
    kvcache: BaseKVCache,
    backend: str,
    page_table: torch.Tensor,
) -> BaseAttnBackend:
    if backend == "auto":
        backend = _resolve_auto_backend(config)
        logger.info(f"Auto-selected attention backend: {backend}")

    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(config, kvcache, p_backend, page_table)
            d_backend = create_attention_backend(config, kvcache, d_backend, page_table)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend

    return SUPPORTED_ATTENTION_BACKENDS[backend](config, kvcache, page_table)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
]
