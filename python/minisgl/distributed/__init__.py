"""Distributed subpackage for minisgl.

The historical eager re-exports (``from .impl import ...``, ``from .info
import ...``) forced ``torch`` and the PyNCCL wrapper to load just to reach
device-agnostic helpers such as :mod:`minisgl.distributed.backend` and
:mod:`minisgl.distributed.runtime`. They now resolve lazily through PEP 562
``__getattr__`` so:

    import minisgl.distributed.backend    # never touches .impl / torch
    import minisgl.distributed.runtime    # never touches .impl / torch

remain safe on Ascend hosts (or a bare macOS dev box) that lack the CUDA
runtime, PyNCCL, or ``torch_npu``. Legacy consumers using
``from minisgl.distributed import DistributedInfo`` etc. continue to work
untouched — the owning submodule executes on first access instead of at
import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed
    from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info

__all__ = [
    "DistributedInfo",
    "get_tp_info",
    "set_tp_info",
    "enable_pynccl_distributed",
    "DistributedCommunicator",
    "try_get_tp_info",
    "destroy_distributed",
]

_LAZY_ATTR_TO_SUBMODULE: dict[str, str] = {
    "DistributedCommunicator": ".impl",
    "destroy_distributed": ".impl",
    "enable_pynccl_distributed": ".impl",
    "DistributedInfo": ".info",
    "get_tp_info": ".info",
    "set_tp_info": ".info",
    "try_get_tp_info": ".info",
}


def __getattr__(name: str) -> Any:
    submod_name = _LAZY_ATTR_TO_SUBMODULE.get(name)
    if submod_name is None:
        raise AttributeError(
            f"module 'minisgl.distributed' has no attribute {name!r}"
        )
    import importlib

    submod = importlib.import_module(submod_name, __name__)
    value = getattr(submod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
