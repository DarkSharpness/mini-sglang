import os
import time
import torch
from minisgl.distributed import set_tp_info
import minisgl.kernel as kernel
import multiprocessing as mp
import pytest


def _run_pynccl_worker(tp_size: int, tp_rank: int):
    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)
    set_tp_info(tp_rank, tp_size)

    # cpu group
    torch.distributed.init_process_group(
        world_size=tp_size,
        rank=tp_rank,
        backend="gloo",
    )

    # use default cpu group
    tp_cpu_group = torch.distributed.group.WORLD
    assert tp_cpu_group is not None, "CPU group should not be None"
    dtype = torch.float16

    K = 512
    USE_SYMM = 0
    N_ITERS = 5

    comm = kernel.init_pynccl(
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=8192 * K * dtype.itemsize if USE_SYMM else 0,
    )

    for i in range(N_ITERS):
        N = 4
        x = torch.ones(8192 * K, dtype=dtype, device=device)
        for _ in range(N):
            comm.all_reduce(x, "sum")
        expected_val = pow(tp_size, N)
        y = torch.full((8192 * K,), expected_val, dtype=dtype, device=device)
        assert torch.allclose(x, y), f"Rank {tp_rank} Iter {i}: Accumulation failed"

        x = torch.full((8192 * K,), tp_rank, dtype=dtype, device=device)
        if tp_rank == 0:
            torch.cuda.synchronize()
            time.sleep(1)
        comm.all_reduce(x, "sum")

        expected_sum = (tp_size * (tp_size - 1)) // 2
        y = torch.full((8192 * K,), expected_sum, dtype=dtype, device=device)
        assert torch.allclose(x, y), f"Rank {tp_rank} Iter {i}: Rank sum failed"

        # to prevent overflow, we use a smaller value for this test
        x = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=device),
                torch.ones((8192 * K // 2,), dtype=dtype, device=device),
            ]
        )
        comm.all_reduce(x, "sum")

        y = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=device),
                torch.full((8192 * K // 2,), tp_size, dtype=dtype, device=device),
            ]
        )
        assert torch.allclose(x, y), f"Rank {tp_rank} Iter {i}: Mixed value failed"

    src = torch.full((K,), tp_rank, dtype=dtype, device=device)
    dst = torch.empty((K * tp_size,), dtype=dtype, device=device)
    torch.cuda.synchronize()
    comm.all_gather(dst, src)
    torch.cuda.synchronize()
    expected = torch.arange(tp_size, dtype=dtype, device=device)
    expected = expected.repeat_interleave(K)
    assert torch.allclose(dst, expected), f"Rank {tp_rank}: All-gather failed"
    torch.distributed.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.distributed
def test_init_pynccl_all_reduce_sum():
    tp_size = 4
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Skipping distributed test: Requires {tp_size} GPUs")
    mp.set_start_method("spawn", force=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    p_list = []
    for i in range(tp_size):
        p = mp.Process(target=_run_pynccl_worker, args=(tp_size, i))
        p_list.append(p)
    try:
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        for p in p_list:
            assert p.exitcode == 0, f"Worker process {p.pid} failed"
    except BaseException:
        for p in p_list:
            p.terminate()
        raise
