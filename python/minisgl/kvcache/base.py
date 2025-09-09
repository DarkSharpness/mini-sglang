from __future__ import annotations

import enum
from abc import ABC, abstractmethod

import torch


class BaseKVCache(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in local LLMs.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...


class KVCacheLayout(enum.Enum):
    LayerFirst = enum.auto()
    PageFirst = enum.auto()
    MixPageLayer = enum.auto()


class KVCacheType(enum.Enum):
    MHA = enum.auto()
