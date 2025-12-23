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


def resolve_auto_backend(config: ModelConfig) -> str:
    """Determine the best attention backend based on the GPU architecture and model."""
    if is_sm100_supported():  # blackwell
        return "fi"
    elif is_sm90_supported():  # hopper
        return "fa,fi"
    else:  # pre-hopper
        return "fi"


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
    from .fi import FlashInferBackend

    return FlashInferBackend(config, kvcache, page_table)


@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config, kvcache, page_table)


def validate_backend(backend: str):
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        supported = SUPPORTED_ATTENTION_BACKENDS.supported_names()
        for b in required_backends:
            if b not in supported:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported attention backend: {b}. Supported backends: {supported}"
                )
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
    kvcache: BaseKVCache,
    page_table: torch.Tensor,
) -> BaseAttnBackend:
    if backend == "auto":
        backend = resolve_auto_backend(config)
        logger.info(f"Auto-selected attention backend: {backend}")

    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config, kvcache, page_table)
            d_backend = create_attention_backend(d_backend, config, kvcache, page_table)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config, kvcache, page_table)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_backend",
]
