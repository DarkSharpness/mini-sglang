from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Literal, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.config.context import Batch, Context, Req, set_global_ctx
from minisgl.distributed import enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers.rotary import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger
from minisgl.utils.torch_utils import torch_dtype

from .config import EngineConfig
from .graph import GraphWorker

logger = init_logger(__name__)


def _get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


@dataclass
class EngineResult:
    next_tokens_cpu: torch.Tensor
    offload_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    onboard_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        assert not torch.cuda.is_initialized()
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        self.tp_cpu_group = self._init_communication()
        free_memory = self._sync_get_memory()[1]
        full_free_memory = free_memory
        logger.info_rank0(f"Free memory before loading model: {free_memory / (1024**3):.2f} GiB")

        # load model and determine number of pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict())
        self.num_pages = self._determine_num_pages(free_memory)

        # initialize core data structures
        self.kv_cache = create_kvcache(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            head_dim=self.model_config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {free_memory / (1024**3):.2f} GiB")

        max_page_entries = config.max_running_req + 1  # +1 for dummy req
        self.page_table = Context.create_page_table(
            max_page_entries, config.max_seq_len, self.device
        )
        self.attn_backend = create_attention_backend(
            config.model_config,
            self.kv_cache,
            config.attention_backend,
            self.page_table,
        )
        self.ctx = Context(
            page_size=1,
            kv_cache=self.kv_cache,
            attn_backend=self.attn_backend,
            page_table=self.page_table,
        )
        set_global_ctx(self.ctx)

        # mapping the dummy req to dummy pages
        self.dummy_req = Req(
            input_ids=[0],
            page_table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            device=self.device,
            uid=-1,
        )
        self.page_table[config.max_running_req].fill_(self.num_pages)

        # cuda graph related
        self.graph_worker = GraphWorker(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=full_free_memory,  # free memory before loading model
            dummy_req=self.dummy_req,
            max_seq_len=config.max_seq_len,
            vocab_size=self.model_config.vocab_size,
        )

        # engine results, use 2 buffers to avoid synchronization
        self.batch_index: Literal[0, 1] = 0
        max_running_req_padded = max(
            config.max_running_req,
            max(config.cuda_graph_bs) if config.cuda_graph_bs else 0,
        )
        self.results = [
            EngineResult(
                next_tokens_cpu=torch.empty(
                    max_running_req_padded,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
            )
            for _ in range(2)
        ]

    def _init_communication(self) -> torch.distributed.ProcessGroup:
        config = self.config
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            if config.use_pynccl:
                max_bytes = (
                    config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
                )
                enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(self.config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int) -> int:
        num_pages, cache_per_page = self._determine_num_pages_impl(old_free_memory)
        assert num_pages > 1, "Not enough memory for KV cache"
        real_size = num_pages * cache_per_page / (1024**3)
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {real_size:.2f} GiB")
        return num_pages

    def _determine_num_pages_impl(self, old_free_memory: int) -> Tuple[int, int]:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, self.config.tp_info.size)
            * self.config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        if self.config.num_page_override is not None:
            return self.config.num_page_override, cache_per_page

        delta = new_free_memory - int(old_free_memory * (1 - self.config.memory_ratio))
        num_pages = delta // cache_per_page
        return num_pages, cache_per_page

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = _get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced: min {min_free_memory / (1024**3):.2f} GB, "
                f"max {max_free_memory / (1024**3):.2f} GB"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _set_input_ids(self, batch: Batch):
        padded_bs = batch.padded_bs
        reqs = batch.reqs + [self.dummy_req] * (padded_bs - batch.batch_size)
        batch.input_ids = torch.cat([req.device_ids[req.cached_len :] for req in reqs])

    def forward_batch(self, batch: Batch):
        assert torch.cuda.current_stream() == self.stream

        # update the input ids only on this stream
        # because the ids is updated only in this stream
        self._set_input_ids(batch)

        with self.ctx.forward_batch(batch):
            if self.graph_worker.can_use_cuda_graph(batch):
                logger.debug_rank0("Using CUDA graph")
                logits = self.graph_worker.replay(batch)
            else:
                logger.debug_rank0("Not using CUDA graph")
                logits = self.model.forward()

        logits = logits[: batch.batch_size]

        # TODO: use a real sampler instead of argmax
        next_tokens = torch.argmax(logits, dim=-1).to(torch.int32)

        # append the next tokens to reqs on GPU stream
        for i, req in enumerate(batch.reqs):
            req.append(next_tokens[i : i + 1])

        # copy next tokens to pinned memory
        self.batch_index = 1 - self.batch_index
        result = self.results[self.batch_index]
        result.next_tokens_cpu[: batch.batch_size].copy_(next_tokens, non_blocking=True)
        result.offload_event.record(self.stream)

    @property
    def last_batch_result(self) -> EngineResult:
        return self.results[self.batch_index]

    def prepare_batch(self, batch: Batch):
        self.attn_backend.prepare_metadata(batch, allow_graph=True)
