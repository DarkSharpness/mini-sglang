from minisgl.kernel_v2 import transfer_hicache
from minisgl.utils import call_if_main, init_logger
import torch

logger = init_logger(__name__)


@call_if_main()
@torch.inference_mode()
def test_hicache_kernel():
    CACHE_ITEM_SIZE = 128
    DTYPE = torch.float16
    CACHE_SIZE = 1024 * 1024
    HOST_CACHE_SIZE = CACHE_SIZE * 2
    BLOCK_QUOTA = 4

    cuda_cache = torch.empty(
        (2, CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cuda",
    )
    host_cache = torch.empty(
        (2, HOST_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cpu",
        pin_memory=True,
    )

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    for bs in [2**n for n in range(5, 18)]:
        indices_dst = torch.randint(0, CACHE_SIZE, (bs,), dtype=torch.int32, device="cuda")
        indices_src = torch.randint(0, HOST_CACHE_SIZE, (bs,), dtype=torch.int32, device="cuda")
        g = torch.cuda.CUDAGraph()
        mem = bs * 2 * CACHE_ITEM_SIZE * cuda_cache.element_size()

        # H -> D
        with torch.cuda.graph(g):
            transfer_hicache(
                k_cache_dst=cuda_cache[0],
                v_cache_dst=cuda_cache[1],
                indices_dst=indices_dst,
                k_cache_src=host_cache[0],
                v_cache_src=host_cache[1],
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
            )

        torch.cuda._sleep(1_000_000_000)
        tic.record()
        for _ in range(100):
            g.replay()

        toc.record()
        toc.synchronize()
        dur = tic.elapsed_time(toc) / 100  # ms
        bandwidth = mem / dur / 1e6  # GB/s
        logger.info(f"H->D bs={bs:6d} time={dur:8.4f} ms bandwidth={bandwidth:8.4f} GB/s")

        indices_dst, indices_src = indices_src, indices_dst  # swap for D->H
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            transfer_hicache(
                k_cache_dst=host_cache[0],
                v_cache_dst=host_cache[1],
                indices_dst=indices_dst,
                k_cache_src=cuda_cache[0],
                v_cache_src=cuda_cache[1],
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
            )
        torch.cuda._sleep(1_000_000_000)
        tic.record()
        for _ in range(100):
            g.replay()
        toc.record()
        toc.synchronize()
        dur = tic.elapsed_time(toc) / 100  # ms
        bandwidth = mem / dur / 1e6  # GB/s
        logger.info(f"D->H bs={bs:6d} time={dur:8.4f} ms bandwidth={bandwidth:8.4f} GB/s")
