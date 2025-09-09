from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal

from .utils import load_kernel_module

if TYPE_CHECKING:
    import torch

    class PyNCCLCommunicator:
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> torch.Tensor: ...
        def all_gather(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor: ...
        def broadcast(self, input: torch.Tensor, src: int) -> torch.Tensor: ...
        def split(self, sizes: List[int]) -> PyNCCLCommunicator | None: ...

else:
    PyNCCLCommunicator = Any


def _load_pynccl_module() -> Any:
    return load_kernel_module("pynccl_2.cu", "pynccl_2_kernel")


def init_pynccl(
    *, tp_rank: int, tp_size: int, tp_cpu_group, max_size_bytes: int | None = None
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
    communicator = module.NCCLWrapper(tp_rank, tp_size, max_size_bytes, nccl_id)
    return communicator
