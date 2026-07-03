import os

import torch
import torch_npu

from minisgl.distributed.runtime import initialize_distributed_from_env


def main() -> None:
    runtime = initialize_distributed_from_env()
    rank = runtime.rank
    world_size = runtime.world_size

    try:
        assert runtime.world_size == 8, (
            f"HCCL smoke test requires world_size=8, "
            f"got {runtime.world_size}"
        )
        assert runtime.device_type == "npu"
        assert runtime.backend == "hccl"
        assert torch.npu.current_device() == runtime.local_rank

        # all-reduce: sum(1..8) = 36
        value = torch.tensor(
            [rank + 1],
            dtype=torch.float32,
            device=runtime.device,
        )
        torch.distributed.all_reduce(value)
        torch.npu.synchronize()
        assert value.item() == world_size * (world_size + 1) / 2

        # broadcast: rank 0 sends 20260704
        payload = torch.tensor(
            [20260704 if rank == 0 else 0],
            dtype=torch.int64,
            device=runtime.device,
        )
        torch.distributed.broadcast(payload, src=0)
        torch.npu.synchronize()
        assert payload.item() == 20260704

        # all-gather: collect rank IDs
        local_rank_id = torch.tensor(
            [rank],
            dtype=torch.int64,
            device=runtime.device,
        )
        gathered = [torch.empty_like(local_rank_id) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, local_rank_id)
        torch.npu.synchronize()
        assert [item.item() for item in gathered] == list(range(world_size))

        torch.distributed.barrier()

        print(
            f"PASS rank={rank} local_rank={runtime.local_rank} "
            f"device={runtime.device} all_reduce={value.item():.0f}",
            flush=True,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
