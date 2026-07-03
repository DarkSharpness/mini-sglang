"""Utility package for minisgl.

Public names are re-exported lazily via PEP 562 ``__getattr__`` so that
submodules like :mod:`minisgl.utils.device` can be imported on hosts that
do not have the full runtime stack (torch, huggingface_hub, msgpack/zmq,
...). This is the pattern required by the Ascend Gate 0.3a contract:

    import minisgl.utils.device          # never pulls in huggingface_hub
    import minisgl.distributed.backend    # never pulls in torch.distributed

Behaviour for legacy callers is preserved — every symbol listed in
``__all__`` remains reachable via ``from minisgl.utils import X`` and
resolves to the exact same object that the eager version would have
returned. The only observable difference is *when* the owning submodule
executes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
    from .hf import cached_load_hf_config, download_hf_weight, load_tokenizer
    from .logger import init_logger
    from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even
    from .mp import (
        ZmqAsyncPullQueue,
        ZmqAsyncPushQueue,
        ZmqPubQueue,
        ZmqPullQueue,
        ZmqPushQueue,
        ZmqSubQueue,
    )
    from .registry import Registry
    from .torch_utils import nvtx_annotate, torch_dtype

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_tokenizer",
    "init_logger",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]

# Attribute → owning submodule. Kept as an explicit table so failures
# surface as clean ``ImportError``s from the real submodule rather than a
# swallowed generic error.
_LAZY_ATTR_TO_SUBMODULE: dict[str, str] = {
    "cached_load_hf_config": ".hf",
    "download_hf_weight": ".hf",
    "load_tokenizer": ".hf",
    "init_logger": ".logger",
    "is_arch_supported": ".arch",
    "is_sm90_supported": ".arch",
    "is_sm100_supported": ".arch",
    "call_if_main": ".misc",
    "div_even": ".misc",
    "div_ceil": ".misc",
    "align_ceil": ".misc",
    "align_down": ".misc",
    "UNSET": ".misc",
    "Unset": ".misc",
    "torch_dtype": ".torch_utils",
    "nvtx_annotate": ".torch_utils",
    "Registry": ".registry",
    "ZmqPushQueue": ".mp",
    "ZmqPullQueue": ".mp",
    "ZmqPubQueue": ".mp",
    "ZmqSubQueue": ".mp",
    "ZmqAsyncPushQueue": ".mp",
    "ZmqAsyncPullQueue": ".mp",
}


def __getattr__(name: str) -> Any:
    submod_name = _LAZY_ATTR_TO_SUBMODULE.get(name)
    if submod_name is None:
        raise AttributeError(f"module 'minisgl.utils' has no attribute {name!r}")
    import importlib

    submod = importlib.import_module(submod_name, __name__)
    value = getattr(submod, name)
    # Cache on the package object so subsequent lookups skip this function.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
