from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal, override

import torch
from minisgl.config.context import Batch, Req
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import CaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


@dataclass
class FICaptureData(CaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    positions:          torch.Tensor  # on gpu
    out_loc:            torch.Tensor  # on gpu
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q:       torch.Tensor  # on gpu
    cu_seqlens_k:       torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    @override
    def get_positions(self) -> torch.Tensor:
        return self.positions

    @override
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="auto",
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer, kv_layout="NHD"
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.dummy_req: Req | None = None
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.page_table = page_table

    def _initialize_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    @override
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_once(metadata)
        self.kvcache.store_kv(k, v, metadata.out_loc, layer_id)
        return metadata.wrapper.run(
            q=q,
            paged_kv_cache=(self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id)),
        )

    @override
    def prepare_metadata(self, batch: Batch, allow_graph: bool, _internal: bool = False) -> None:
        from minisgl.kernel import load_decode_indices

        given_bs = len(batch.reqs)
        reqs = batch.reqs.copy()

        # if we can use the cuda graph, pad the reqs to the next available bs
        # since batch is not complete (no metadata yet), we can't use `can_use_graph` here
        if (
            allow_graph
            and self.capture is not None
            and batch.is_decode
            and given_bs <= self.max_graph_bs
        ):
            assert self.dummy_req is not None
            next_bs = next(bs for bs in self.capture_bs if bs >= given_bs)
            reqs += [self.dummy_req] * (next_bs - given_bs)
        padded_bs = len(reqs)
        del given_bs

        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {
            "device": "cpu",
            "dtype": torch.int32,
            "pin_memory": True,
        }

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        if max_seqlen_q == 1:
            cu_seqlens_q_cpu = torch.arange(0, padded_bs + 1, **cpu_kwargs)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)

        page_table = self.page_table

        if _internal:
            # will be set later in `prepare_for_capture`
            out_loc = torch.tensor([])
            positions = torch.tensor([])
        else:
            if max_seqlen_q == 1:
                # we may benefit from lru cache here; even not, this is still faster than the below
                out_loc = load_decode_indices(
                    page_table=page_table,
                    pos=[(req.page_table_idx, req.cached_len) for req in reqs],
                )
            else:
                out_loc = torch.cat(
                    [
                        page_table[req.page_table_idx, req.cached_len : req.device_len]
                        for req in reqs
                    ]
                )

            positions = (
                torch.cat(
                    [torch.arange(i, j, dtype=torch.int32) for i, j in zip(cached_lens, seqlens_k)]
                )
                .pin_memory()
                .to(device, non_blocking=True)
            )

        tp_size = get_tp_info().size
        qo_head_local = divide_even(self.config.num_qo_heads, tp_size)
        kv_head_local = divide_even(self.config.num_kv_heads, tp_size)

        # copy from CPU to GPU
        batch.attn_metadata = FIMetadata(
            positions=positions,
            out_loc=out_loc,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_k=cu_seqlens_k_cpu.to(device, non_blocking=True),
            cu_seqlens_q=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.page_table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_bs),
            num_qo_heads=qo_head_local,
            num_kv_heads=kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )
        batch.padded_bs = padded_bs

    @override
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], dummy_req: Req) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        cuda_int_kwargs = {"device": self.kvcache.device, "dtype": torch.int32}
        capture = FICaptureData(
            input_ids=torch.zeros(max_bs, **cuda_int_kwargs),
            seq_lens=torch.ones(max_bs, **cuda_int_kwargs),
            positions=torch.zeros(max_bs, **cuda_int_kwargs),
            cu_seqlens_k=torch.arange(0, max_bs + 1, **cuda_int_kwargs),
            cu_seqlens_q=torch.arange(0, max_bs + 1, **cuda_int_kwargs),
            page_table=torch.zeros((max_bs * max_seq_len), **cuda_int_kwargs),
            out_loc=torch.zeros(max_bs, **cuda_int_kwargs),
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.dummy_req = dummy_req
        self.capture_bs = sorted(bs_list)
        assert dummy_req.extend_len == 1, "Dummy req must be for decode."

    @cached_property
    def use_tensor_cores(self) -> bool:
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    @override
    def prepare_for_capture(self, bs: int) -> Batch:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        assert bs in self.capture_bs and self.capture and self.dummy_req
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")

        assert (
            bs not in self.graph_wrappers
        ), f"Graph for bs={bs} already captured, {self.graph_wrappers.keys()=}"
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer

        self.prepare_metadata(batch, allow_graph=False, _internal=True)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        metadata.out_loc = capture.out_loc[:bs]
        metadata.positions = capture.positions[:bs]
        batch.input_ids = capture.input_ids[:bs]
        self._initialize_once(metadata)
        return batch

    def _copy_metadata(self, metadata: FIMetadata, input_ids: torch.Tensor, bs: int) -> None:
        assert self.capture is not None and bs in self.capture_bs
        assert len(input_ids) == bs <= self.max_graph_bs
        self.capture.input_ids[:bs].copy_(input_ids)
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.out_loc[:bs].copy_(metadata.out_loc)

    @override
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        self._copy_metadata(metadata, batch.input_ids, batch.padded_bs)
        metadata.wrapper = self.graph_wrappers[batch.padded_bs]
        self._initialize_once(metadata)
