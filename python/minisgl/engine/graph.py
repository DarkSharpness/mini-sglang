from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    table_indices: torch.Tensor # GPU-native indices to index into the VRAM ledger of per-request hidden states
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            table_indices=torch.zeros(bs, dtype=torch.int64, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        # used during graph capture: binds the batch data to the buffer's static memory addresses
        # so the graph records the correct pointers
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.table_indices = self.table_indices[_slice]

    def copy_from(self, batch: Batch) -> None:
        # used during graph replay: copies the dynamic batch data into the static GPU buffer before launcing the CUDA graph
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        self.table_indices[_slice] = batch.table_indices


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
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        
        # num blocks to dynamically capture nested graphs
        self.num_blocks = len(model.model.blocks) 
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # nested dictionary to map: batch_size -> block_idx -> graph_type
        self.graph_map: Dict[int, Dict[int, Dict[str, torch.cuda.CUDAGraph]]] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            
            self.graph_map[bs] = {} # initialise block map for this batch size
            
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            
            # set dummy indices for the VRAM ledger so the graph records the scatter/gather
            batch.table_indices = torch.full((bs,), self.dummy_req.table_idx, dtype=torch.int64, device=self.device)
            
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)

            # loop through each block to capture its isolated graphs
            for block_idx in range(self.num_blocks):
                self.graph_map[bs][block_idx] = {}

                # capture the standard full-compute graph
                graph_compute = torch.cuda.CUDAGraph()
                with get_global_ctx().forward_batch(batch):
                    # warmup pass
                    logits = model.forward_single_block(block_idx, is_project=False, batch=batch)
                    if logits is not None: self.buffer.logits[:bs] = logits
                    
                    # record pass
                    with torch.cuda.graph(graph_compute, pool=pool, stream=self.stream):
                        logits = model.forward_single_block(block_idx, is_project=False, batch=batch)
                        if logits is not None: self.buffer.logits[:bs] = logits
                        
                if pool is None:
                    pool = graph_compute.pool()
                self.graph_map[bs][block_idx]["compute"] = graph_compute

                # capture the project-only, skipping graph
                graph_project = torch.cuda.CUDAGraph()
                with get_global_ctx().forward_batch(batch):
                    # warmup pass
                    model.forward_single_block(block_idx, is_project=True, batch=batch)
                    
                    # record pass
                    with torch.cuda.graph(graph_project, pool=pool, stream=self.stream):
                        model.forward_single_block(block_idx, is_project=True, batch=batch)
                self.graph_map[bs][block_idx]["project"] = graph_project

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    # replay takes in block_idx and is_project to select the exact sub-graph
    def replay(self, batch: Batch, block_idx: int, is_project: bool) -> torch.Tensor | None:
        g_type = "project" if is_project else "compute"
        
        assert self.can_use_cuda_graph(batch)

        # if we are running the first block, we need to prepare the batch and attention metadata
        if block_idx == 0:
            self.buffer.copy_from(batch)
            self.attn_backend.prepare_for_replay(batch)
        
        g = self.graph_map[batch.padded_size][block_idx][g_type]
        g.replay()
        
        # only return logits if this is the final compute block of the model
        # even if is_project is True, the final block's project graph will still produce logits that we want to return
        if block_idx == self.num_blocks - 1:
            return self.buffer.logits[: batch.size]
        return None

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
