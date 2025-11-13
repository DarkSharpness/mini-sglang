from __future__ import annotations

from typing import Any, Callable, Dict

from minisgl.utils import init_logger

logger = init_logger(__name__)


def perf_cuda(
    f: Callable[[], Any],
    *,
    init_stream: bool = True,
    repetitions: int = 10,
    cuda_graph_repetitions: int | None = 10,
) -> float:
    import torch

    assert repetitions > 0
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if init_stream:
        stream = torch.cuda.Stream()
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        f()
        if N := cuda_graph_repetitions:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(N):
                    f()
            replay = g.replay
            del g
        else:
            replay = f
            N = 1

        torch.cuda.synchronize()

        replay()
        tic.record()
        for _ in range(repetitions):
            replay()
        toc.record()
        toc.synchronize()
        dur = tic.elapsed_time(toc)
        return dur / (N * repetitions)


def compare_memory_kernel_perf(
    *,
    our_impl: Callable[[], Any],
    baseline: Callable[[], Any],
    memory_footprint: int,  # in bytes
    prefix_msg: str = " ",
    extra_kwargs: Dict[str, Any] | None = None,
) -> None:
    dur = perf_cuda(baseline, **(extra_kwargs or {}))
    bandwidth = memory_footprint / (dur * 1e6)  # GB/s
    logger.info(f"{prefix_msg}Baseline: {dur:8.3f} ms | {bandwidth:8.3f} GB/s")

    dur = perf_cuda(our_impl, **(extra_kwargs or {}))
    bandwidth = memory_footprint / (dur * 1e6)  # GB/s
    logger.info(f"{prefix_msg}Our Impl: {dur:8.3f} ms | {bandwidth:8.3f} GB/s")
