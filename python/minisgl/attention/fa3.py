from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, override

import torch
from minisgl.config.context import Batch, Req

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import CaptureData

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


@dataclass
class FA3CaptureData(CaptureData):
    pass


@dataclass
class FA3Metadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    positions: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    out_loc: torch.Tensor
    page_table: torch.Tensor

    def get_positions(self) -> torch.Tensor:
        return self.positions

    @override
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        self.config = config
        self.kvcache = kvcache
        self.capture: FA3CaptureData | None = None
        self.dummy_req: Req | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.page_table = page_table

    @override
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FA3Metadata)
        self.kvcache.store_kv(k, v, metadata.out_loc, layer_id)
        return _fa3_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
        )

    @override
    def prepare_metadata(self, batch: Batch, allow_graph: bool) -> None:
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
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {
            "device": "cpu",
            "dtype": torch.int32,
            "pin_memory": True,
        }

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **cpu_kwargs)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_bs + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        positions = (
            torch.cat(
                [torch.arange(i, j, dtype=torch.int32) for i, j in zip(cached_lens, seqlens_k)]
            )
            .pin_memory()
            .to(device, non_blocking=True)
        )

        page_table = self.page_table

        if max_seqlen_q == 1:
            # we may benefit from lru cache here; even not, this is still faster than the below
            out_loc = load_decode_indices(
                page_table=page_table,
                pos=[(req.page_table_idx, req.cached_len) for req in reqs],
            )
        else:
            out_loc = torch.cat(
                [page_table[req.page_table_idx, req.cached_len : req.device_len] for req in reqs]
            )

        new_page_table = torch.stack(
            [page_table[req.page_table_idx, :max_seqlen_k] for req in reqs]
        )

        # copy from CPU to GPU
        batch.attn_metadata = FA3Metadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            positions=positions,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            out_loc=out_loc,
            page_table=new_page_table,
        )
        batch.padded_bs = padded_bs

    @override
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], dummy_req: Req) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        cuda_int_kwargs = {"device": self.kvcache.device, "dtype": torch.int32}
        capture = FA3CaptureData(
            input_ids=torch.zeros(max_bs, **cuda_int_kwargs),
            seq_lens=torch.ones(max_bs, **cuda_int_kwargs),
            positions=torch.zeros(max_bs, **cuda_int_kwargs),
            cu_seqlens_k=torch.arange(0, max_bs + 1, **cuda_int_kwargs),
            cu_seqlens_q=torch.arange(0, max_bs + 1, **cuda_int_kwargs),
            page_table=torch.zeros((max_bs, max_seq_len), **cuda_int_kwargs),
            out_loc=torch.zeros(max_bs, **cuda_int_kwargs),
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.dummy_req = dummy_req
        self.capture_bs = sorted(bs_list)
        assert dummy_req.extend_len == 1, "Dummy req must be for decode."

    @override
    def prepare_for_capture(self, bs: int) -> Batch:
        assert bs in self.capture_bs and self.capture and self.dummy_req
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")

        capture = self.capture
        metadata = FA3Metadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            positions=capture.positions[:bs],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1),  # maximum seqlen k
            max_seqlen_q=1,  # decode only
            out_loc=capture.out_loc[:bs],
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.padded_bs = bs
        return batch

    def _copy_metadata(self, metadata: FA3Metadata, input_ids: torch.Tensor, bs: int) -> None:
        assert self.capture is not None and bs in self.capture_bs
        assert len(input_ids) == bs <= self.max_graph_bs
        self.capture.input_ids[:bs].copy_(input_ids)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, : metadata.max_seqlen_k].copy_(metadata.page_table)
        self.capture.out_loc[:bs].copy_(metadata.out_loc)

    @override
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FA3Metadata)
        self._copy_metadata(metadata, batch.input_ids, batch.padded_bs)


def _fa3_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    o: torch.Tensor | None = None,  # Can be used to save memory
) -> torch.Tensor:
    try:
        import sgl_kernel.flash_attn  # noqa: F401
    except ImportError:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`."
        )

    for x in (k_cache, v_cache, q, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k_new):
        assert x.stride(-1) == 1, "this tensor must have contiguous last dimension"

    out, *_ = torch.ops.sgl_kernel.fwd.default(  # type: ignore
        q,
        k_cache,
        v_cache,
        None,  # k
        None,  # v
        None,  # q_v,
        o,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        None,  # kv_batch_idx_,
        None,  # leftpad_k_,
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # rotary_seqlens
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        softmax_scale,
        True,  # causal
        window_size[0],
        window_size[1],
        softcap,
        True,  # rotary_interleaved
        None,  # scheduler_metadata
        num_splits,
        pack_gqa,
        sm_margin,
        None,  # q_v_descale
    )

    return out
