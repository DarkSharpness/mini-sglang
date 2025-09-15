from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, override

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.kernel import PyNCCLCommunicator

    from .info import DistributedInfo


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_buffer(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    @override
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @override
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out

    @override
    def get_buffer(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.cuda.current_device()
        return torch.empty_like(x, device=f"cuda:{device}")


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    @override
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.comm.all_reduce(x, "sum")

    @override
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.comm.all_gather(x)

    @override
    def get_buffer(self, x: torch.Tensor) -> torch.Tensor:
        return self.comm.get_buffer(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    def get_buffer(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].get_buffer(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
