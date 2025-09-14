from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .utils import load_kernel_module

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> torch.Tensor: ...
        @abstractmethod
        def all_gather(self, input: torch.Tensor) -> torch.Tensor: ...
        @abstractmethod
        def get_buffer(self, input: torch.Tensor) -> torch.Tensor: ...

else:
    PyNCCLCommunicator = Any


def _load_pynccl_module() -> Any:
    return load_kernel_module("pynccl.cu", "pynccl_wrapper")


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
    allow_fallback: bool = True,
    n_buf: int = 3,
    device: torch.device | None = None,
) -> PyNCCLCommunicator:
    import torch

    module = _load_pynccl_module()
    if tp_rank == 0:
        id_list = [module.get_nccl_unique_id()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"
    if device is None:
        device = torch.device("cuda", tp_rank)
    communicator = module.NCCLWrapper(
        tp_rank, tp_size, max_size_bytes, nccl_id, device, allow_fallback, n_buf
    )
    return communicator
