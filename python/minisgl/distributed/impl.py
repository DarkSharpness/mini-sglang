from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator

ReduceOp = Literal["sum", "prod", "max", "min", "avg"]

TORCH_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "prod": dist.ReduceOp.PRODUCT,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "avg": dist.ReduceOp.AVG,
}


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor, op: ReduceOp = "sum") -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor, op: ReduceOp = "sum") -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        reduce_op = TORCH_REDUCE_OPS.get(op)
        if reduce_op is None:
            raise ValueError(f"Unsupported reduce op: {op}")
        dist.all_reduce(x, op=reduce_op)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor, op: ReduceOp = "sum") -> torch.Tensor:
        self.comm.all_reduce(x, op)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor, op: ReduceOp = "sum") -> torch.Tensor:
        return self.plugins[-1].all_reduce(x, op)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
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


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
