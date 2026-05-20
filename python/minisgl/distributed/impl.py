from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
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

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
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

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


_EP_GROUP: dist.ProcessGroup | None = None


def set_ep_group(group: dist.ProcessGroup) -> None:
    global _EP_GROUP
    _EP_GROUP = group


def get_ep_group() -> dist.ProcessGroup:
    assert _EP_GROUP is not None, "EP group has not been initialized"
    return _EP_GROUP


def ep_all_to_all(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
) -> None:
    dist.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=_EP_GROUP,
    )


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
    global _EP_GROUP
    DistributedCommunicator.plugins = []
    _EP_GROUP = None
