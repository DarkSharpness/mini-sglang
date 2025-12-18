from __future__ import annotations

import gc
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
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


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ):
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        if len(cuda_graph_bs) == 0:
            logger.info_rank0("CUDA graph is disabled.")
            self.max_graph_bs = 0
            self.graph_map = {}
            return

        cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
        self.max_graph_bs = max(cuda_graph_bs)
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float16,
            device=device,
        )
        self.attn_backend = attn_backend
        attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=cuda_graph_bs)

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        free_memory = get_free_memory(device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
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
            free_memory = get_free_memory(device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            g = torch.cuda.CUDAGraph()
            if bs != self.max_graph_bs:
                batch = Batch(reqs=[dummy_req] * bs, phase="decode")
                self.attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                self.logits[:bs] = model.forward()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.logits[:bs] = model.forward()
            if pool is None:
                pool = g.pool()
            graph_list.append((bs, g))

        free_memory = get_free_memory(device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

        self.graph_map = dict(graph_list)
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.logits[: batch.size]

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()

    def pad_batch(self, batch: Batch) -> int:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size
