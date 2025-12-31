import multiprocessing as mp
import os

import minisgl.kernel as kernel
import torch
from minisgl.distributed import set_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

logger = init_logger(__name__)


@torch.no_grad()
def run_benchmark(tp_size: int, tp_rank: int):
    torch.cuda.set_device(tp_rank)
    torch.cuda.set_stream(torch.cuda.Stream(tp_rank))  # type: ignore
    stream = torch.cuda.current_stream()
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

    comm = kernel.init_pynccl(
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=8192 * K * dtype.itemsize if USE_SYMM else 0,
    )

    def bench_performance(f, use_graph=False):
        import gc

        gc.collect()
        gc.disable()

        N = 1024
        M = 16
        x = torch.zeros(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
        f(x)
        f(x)

        pbar = tqdm(list(range(N)), desc="Capturing cuda graph", disable=tp_rank > 0)

        torch.cuda.synchronize()
        if use_graph:
            g = torch.cuda.CUDAGraph()
            graph = torch.cuda.graph(g)
            with graph:
                for _ in pbar:
                    f(x)
            cur_stream = graph.capture_stream
        else:
            nonlocal stream
            f(x)
            f(x)
            cur_stream = stream

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(cur_stream):
            tic.record(cur_stream)
            if use_graph:
                for _ in range(M):
                    g.replay()  # type: ignore
            else:
                for _ in range(M):
                    for _ in pbar:
                        f(x)
            toc.record(cur_stream)
        gc.enable()
        toc.synchronize()
        elapsed_time = tic.elapsed_time(toc)
        avg_time = elapsed_time * 1000 / (M * N)
        logger.info(f"Rank {tp_rank} all-reduce avg time: {avg_time: .4f} us")
        bandwidth = (8192 * K * dtype.itemsize) / (avg_time * 1e3)  # in GB/s
        logger.info(f"Rank {tp_rank} all-reduce bandwidth: {bandwidth:.2f} GB/s")
        # print memory usage
        mem_usage = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(f"Rank {tp_rank} memory usage: {mem_usage:.2f} MB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    bench_performance(lambda x: comm.all_reduce(x, "sum"))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    tp_size = 4
    mp.set_start_method("spawn", force=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    p_list = []
    for i in range(tp_size):
        p = mp.Process(target=run_benchmark, args=(tp_size, i))
        p_list.append(p)
    try:
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
    except BaseException:
        for p in p_list:
            p.terminate()
        raise
