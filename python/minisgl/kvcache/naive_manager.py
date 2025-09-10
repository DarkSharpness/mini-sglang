from __future__ import annotations

from typing import Tuple, final, override

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    pass


@final
class NaiveCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()

    @override
    def match_prefix_len(self, input_ids: torch.Tensor) -> int:
        return 0

    @override
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[NaiveCacheHandle, torch.Tensor]:
        _ = input_ids  # unused
        return NaiveCacheHandle(), self.empty_tensor

    @override
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        _ = handle, unlock  # unused

    @override
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        assert len(indices) == len(input_ids)
        return len(indices)

    @override
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    @override
    def reset(self) -> None:
        pass

    @property
    @override
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    @override
    def check_integrity(self) -> None:
        pass
