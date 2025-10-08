from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.config.context import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.config.context import Req
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


class GraphWorker:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        dummy_req: Req,
        max_seq_len: int,
        vocab_size: int,
    ):
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        if len(cuda_graph_bs) == 0:
            logger.info_rank0("CUDA graph is disabled.")
            self.max_graph_bs = 0
            return

        cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
        self.max_graph_bs = max(cuda_graph_bs)
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float16,
            device=device,
        )
        self.attn_backend = attn_backend
        attn_backend.init_capture_graph(
            max_seq_len=max_seq_len,
            bs_list=cuda_graph_bs,
            dummy_req=dummy_req,
        )

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        free_memory = torch.cuda.mem_get_info(device)[0]
        logger.info_rank0(
            f"Free GPU memory before capturing CUDA graphs: {free_memory / (1 << 30):.2f} GiB"
        )

        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = attn_backend.prepare_for_capture(self.max_graph_bs)
        with get_global_ctx().forward_batch(batch):
            self.logits[:] = model.forward()
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        del g

        graph_list: List[Tuple[int, torch.cuda.CUDAGraph]] = []
        pbar = tqdm(
            cuda_graph_bs,
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )

        pool = None
        for bs in pbar:
            remaining_memory, _ = torch.cuda.mem_get_info(device)
            pbar.desc = (
                "Capturing graphs: "
                f"bs = {bs:<3} | "
                f"avail_mem = {remaining_memory / (1 << 30):.2f} GiB"
            )
            pbar.refresh()
            g = torch.cuda.CUDAGraph()
            if bs != self.max_graph_bs:
                batch = attn_backend.prepare_for_capture(bs)
            with get_global_ctx().forward_batch(batch):
                self.logits[:bs] = model.forward()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.logits[:bs] = model.forward()
            if pool is None:
                pool = g.pool()
            graph_list.append((bs, g))

        free_memory = torch.cuda.mem_get_info(device)[0]
        logger.info_rank0(
            f"Free GPU memory after capturing CUDA graphs: {free_memory / (1 << 30):.2f} GiB"
        )

        # Sort by batch size ascendingly for easy searching
        self.graph_list = sorted(graph_list, key=lambda x: x[0])

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.batch_size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if batch.batch_size != batch.padded_bs:
            logger.debug_rank0(f"Padding from {batch.batch_size} to {batch.padded_bs}")
        g = next(g for bs, g in self.graph_list if bs == batch.padded_bs)
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.logits[: batch.batch_size]
