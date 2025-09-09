from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.config.context import Batch, Context, Req, set_global_ctx
from minisgl.config.engine import EngineConfig
from minisgl.distributed import enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers.rotary import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger
from minisgl.utils.torch_utils import torch_dtype

logger = init_logger(__name__)


def _get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


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
        self.attn_backend = create_attention_backend(self.kv_cache, config.attention_backend)
        self.ctx = Context(
            page_num=self.num_pages,
            page_size=1,
            max_running_req=config.max_running_req + 1,  # +1 for dummy req
            max_seq_len=config.max_seq_len,
            device=self.device,
            kv_cache=self.kv_cache,
            attn_backend=self.attn_backend,
        )
        set_global_ctx(self.ctx)

        # mapping the dummy req to dummy pages
        self.dummy_req = Req(
            input_ids=[0],
            page_table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            device=self.device,
            rid=-1,
        )
        self.page_table = self.ctx.page_table
        assert len(self.page_table) == config.max_running_req + 1
        self.page_table[config.max_running_req].fill_(self.num_pages)

    def _init_communication(self) -> torch.distributed.ProcessGroup:
        config = self.config
        if config.use_pynccl:
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
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
            return {k: torch.randn_like(v) for k, v in self.model.state_dict().items()}
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(self.config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int) -> int:
        num_pages, cache_per_page = self._determine_num_pages_impl(old_free_memory)
        assert num_pages > 1, "Not enough memory for KV cache"
        real_size = num_pages * cache_per_page / (1 << 30)
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

    def forward_batch(self, batch: Batch, finalize: bool = False) -> torch.Tensor:
        assert torch.cuda.current_stream() == self.stream
        if finalize:
            batch.attn_metadata.finalize(self.ctx.page_table)
        with self.ctx.forward_batch(batch):
            logits = self.model.forward()
        return logits

    def prepare_batch(self, batch: Batch, finalize: bool = True):
        self.attn_backend.prepare_metadata(batch, allow_graph=False)
        if finalize:
            batch.attn_metadata.finalize(self.ctx.page_table)
