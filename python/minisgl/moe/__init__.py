from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend() -> BaseMoeBackend:
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("ep")
def create_ep_moe_backend() -> BaseMoeBackend:
    from .ep import EPMoe

    return EPMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
