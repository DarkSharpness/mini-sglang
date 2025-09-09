from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, final

import torch
import torch.distributed as dist
from minisgl.distributed import DistributedInfo

if TYPE_CHECKING:
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...


@final
@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, out: torch.Tensor, x: torch.Tensor):
        dist.all_gather_into_tensor(out, x)
        return out


@final
@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.comm.all_reduce(x, "sum")

    def all_gather(self, out: torch.Tensor, x: torch.Tensor):
        return self.comm.all_gather(x, out)


_GLOBAL_PLUGINS: List[DistributedImpl] = [TorchDistributedImpl()]


def enable_pynccl_distributed(tp_info: DistributedInfo, tp_cpu_group, max_bytes: int) -> None:
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    _GLOBAL_PLUGINS.append(PyNCCLDistributedImpl(comm))


def get_distributed_impl() -> DistributedImpl:
    return _GLOBAL_PLUGINS[-1]
