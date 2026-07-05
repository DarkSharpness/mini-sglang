"""Ascend Fused Infer Attention (FIA) backend — Gate 1.8c metadata builder.

Gate 1.8a landed the class scaffolding. Gate 1.8c fills in the metadata
constructor for the single-request (B=1) BSND path only — multi-request
TND, NPU-graph capture wiring, and the FIA call itself remain deferred.

The module stays torch-free at import time: ``torch`` and ``minisgl.core``
are pulled in lazily inside the metadata methods so that importing
``minisgl.attention.ascend_fia`` on a CUDA / CPU host is safe. The module
must never reference ``torch_npu`` (top-level or lazy) — the FIA op wiring
lands in a later Gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch
    from minisgl.models import ModelConfig


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class FIAMetadata(BaseAttnMetadata):
    """Metadata for the single-request BSND Ascend FIA path.

    The multi-request TND path and the NPU-graph replay path will extend
    this dataclass in later Gates; today we only surface the fields needed
    for B=1 prefill / decode.

    Fields:
        block_table:            [1, num_blocks] int32 on the KV-cache device;
                                each entry is a physical **page id** (not raw
                                slot) — matches the layout FIA's
                                ``block_table`` argument expects.
        actual_seq_lengths:     per-request query length; a Python list
                                (length == 1 today) so it can be passed
                                straight to ``torch_npu.npu_fused_infer_...``
                                without a device round-trip.
        actual_seq_lengths_kv:  per-request total KV length (cached + extend).
        input_layout:           ``"BSND"`` for the single-request path.
        query_seq_len:          == ``actual_seq_lengths[0]``, kept for cheap
                                indexing.
        kv_seq_len:             == ``actual_seq_lengths_kv[0]``.
    """

    block_table: "torch.Tensor"
    actual_seq_lengths: List[int]
    actual_seq_lengths_kv: List[int]
    input_layout: str
    query_seq_len: int
    kv_seq_len: int

    def get_last_indices(self, bs: int) -> "torch.Tensor":
        """Return the index of the last query token per request.

        For the single-request path this is a 1-element tensor:

        * prefill: ``query_seq_len - 1`` (last new token in the flat Q buffer)
        * decode:  ``0`` (``query_seq_len == 1``)

        Multi-request wiring lands with the TND path; ``bs != 1`` is rejected
        explicitly rather than silently returning a mis-shaped tensor.
        """
        if bs != 1:
            raise NotImplementedError(
                "Ascend FIA get_last_indices currently supports batch size 1 only"
            )
        # Lazy import so the module stays torch-free at import time.
        import torch

        return torch.tensor(
            [self.query_seq_len - 1],
            dtype=torch.int32,
            device=self.block_table.device,
        )


class AscendFIABackend(BaseAttnBackend):
    """Skeleton for the Ascend FIA paged-KV attention backend.

    The constructor signature mirrors what :func:`create_attention_backend`
    already passes to :class:`FlashInferBackend` / :class:`FlashAttentionBackend`
    / :class:`TensorRTLLMBackend`: a single ``ModelConfig`` positional. We only
    stash the config for later gates; no NPU state is materialised yet.
    """

    def __init__(self, config: "ModelConfig") -> None:
        self.config = config

    # --------------------------------------------------------------- graph
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # Real NPU-graph capture wiring lands with the FIA call in a later Gate.
        return None

    def prepare_for_capture(self, batch: "Batch") -> None:
        return None

    def prepare_for_replay(self, batch: "Batch") -> None:
        return None

    # ------------------------------------------------------------ metadata
    def prepare_metadata(self, batch: "Batch") -> None:
        """Build :class:`FIAMetadata` for the single-request BSND path.

        Only ``len(batch.padded_reqs) == 1`` is supported today. The
        multi-request TND and NPU-graph decode paths are separate Gates.
        """
        reqs = batch.padded_reqs
        if len(reqs) != 1:
            raise NotImplementedError(
                "Ascend FIA metadata currently supports batch size 1 only"
            )

        # Lazy imports keep the module import-safe on CUDA / CPU hosts and
        # avoid pulling ``minisgl.core`` (which imports torch) at registration
        # time.
        import torch
        from minisgl.core import get_global_ctx

        req = reqs[0]
        query_seq_len = req.extend_len
        kv_seq_len = req.device_len

        ctx = get_global_ctx()
        page_table = ctx.page_table
        page_size = ctx.page_size
        num_blocks = _ceil_div(kv_seq_len, page_size)

        # The global page_table stores raw slots (page_id * page_size + offset).
        # We stride by ``page_size`` to pick the first slot of each page, then
        # divide to recover the physical page id. Stride-then-divide matches
        # the pattern already used by ``FlashAttentionBackend`` (fa.py:93) and
        # ``TensorRTLLMBackend`` (trtllm.py:117). ``num_blocks * page_size`` is
        # the stop bound (exclusive) — safe because the row is allocated to
        # ``aligned_max_seq_len >= num_blocks * page_size``.
        raw_slots = page_table[
            req.table_idx,
            : num_blocks * page_size : page_size,
        ]
        block_table = (raw_slots // page_size).view(1, num_blocks)

        # Structural invariants FIA relies on. Cheap enough to keep at runtime.
        assert block_table.shape == (1, num_blocks), (
            f"expected block_table shape (1, {num_blocks}), got {tuple(block_table.shape)}"
        )
        assert block_table.dtype == torch.int32, (
            f"expected block_table dtype int32, got {block_table.dtype}"
        )
        assert block_table.device == page_table.device, (
            f"expected block_table on {page_table.device}, got {block_table.device}"
        )

        batch.attn_metadata = FIAMetadata(
            block_table=block_table,
            actual_seq_lengths=[query_seq_len],
            actual_seq_lengths_kv=[kv_seq_len],
            input_layout="BSND",
            query_seq_len=query_seq_len,
            kv_seq_len=kv_seq_len,
        )

    # ------------------------------------------------------------- forward
    def forward(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        layer_id: int,
        batch: "Batch",
    ) -> "torch.Tensor":
        """Run one paged-KV FIA attention step (single-request BSND).

        Execution order (mirrors the spec for Gate 1.8e):

        1. validate ``batch.attn_metadata`` is a :class:`FIAMetadata`;
        2. re-check we're still on the single-request path (defense-in-depth
           against a caller that mutated ``padded_reqs`` between
           :meth:`prepare_metadata` and here);
        3. write this layer's new K/V into the paged cache;
        4. no-copy view the flat query ``[Tq, Hq, D]`` as BSND
           ``[1, Tq, Hq, D]``;
        5. fetch the FIA-native BnNBsD cache tensors — passed verbatim,
           no permute / contiguous;
        6. build the causal ``atten_mask`` (or ``None`` for the single-Q
           case) padded to the block boundary FIA's tiling requires;
        7. dynamic-import ``torch_npu`` and call
           :func:`torch_npu.npu_fused_infer_attention_score`;
        8. take the first tensor of the returned tuple (softmax_lse is
           unused in inference mode);
        9. no-copy view back to the flat ``[Tq, Hq, D]`` shape.
        """
        # 1. metadata type check
        metadata = batch.attn_metadata
        if not isinstance(metadata, FIAMetadata):
            raise TypeError(
                "Ascend FIA forward expects batch.attn_metadata to be "
                f"FIAMetadata, got {type(metadata).__name__}"
            )
        # 2. still B=1
        if len(metadata.actual_seq_lengths) != 1:
            raise NotImplementedError(
                "Ascend FIA forward currently supports batch size 1 only"
            )

        # Lazy imports keep the module import-safe on CUDA / CPU hosts.
        import torch
        from minisgl.core import get_global_ctx

        ctx = get_global_ctx()

        # 3. write this layer's new K/V into the paged cache. store_kv covers
        # both nhd and bnbsd layouts internally; the FIA path is always bnbsd
        # so store_kv writes directly with the raw-slot scatter from Gate 1.7f.
        ctx.kv_cache.store_kv(k, v, batch.out_loc, layer_id)

        # 4. no-copy view [Tq, Hq, D] -> [1, Tq, Hq, D]. unsqueeze is a
        # guaranteed non-copy view regardless of contiguity.
        query_bsnd = q.unsqueeze(0)

        # 5. BnNBsD paged caches — passed verbatim, no permute/contiguous.
        key_cache = ctx.kv_cache.k_cache(layer_id)
        value_cache = ctx.kv_cache.v_cache(layer_id)

        # 6. Causal mask. Decode / single-Q path uses ``None`` (FIA elides
        # masking). Prefill builds an offset causal mask, then pads the KV
        # axis to ``num_blocks * page_size`` because FIA's tiling requires
        # ``atten_mask.shape[-1] >= num_blocks * block_size`` — the extra
        # columns are all masked out (``True``) so they can never
        # participate. This constraint was surfaced by the Gate 1.8d probe.
        if metadata.query_seq_len == 1:
            atten_mask = None
        else:
            padded_kv_len = metadata.block_table.shape[1] * ctx.page_size
            cached_len = metadata.kv_seq_len - metadata.query_seq_len
            q_pos = cached_len + torch.arange(
                metadata.query_seq_len, device=q.device
            )
            k_pos = torch.arange(padded_kv_len, device=q.device)
            atten_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)

        # 7. Dynamic import of torch_npu. Gate 1.8a forbids this at module
        # top level; only here inside forward() is it allowed. Surface a
        # clean RuntimeError so downstream operators aren't left staring at
        # a bare ImportError.
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError(
                "Ascend FIA forward requires torch_npu to be importable; "
                "install torch_npu on the host to use the 'npu_fia' "
                "attention backend."
            ) from exc

        # 8. Call FIA. ``scale`` — not ``scale_value`` — matches the aclnn v3
        # binding. ``num_heads`` is Hq from the flat query (q.shape[1]);
        # ``num_key_value_heads`` is Hkv from the paged cache
        # (key_cache.shape[1] under BnNBsD).
        result = torch_npu.npu_fused_infer_attention_score(
            query_bsnd,
            key_cache,
            value_cache,
            atten_mask=atten_mask,
            actual_seq_lengths=metadata.actual_seq_lengths,
            actual_seq_lengths_kv=metadata.actual_seq_lengths_kv,
            block_table=metadata.block_table,
            num_heads=q.shape[1],
            num_key_value_heads=key_cache.shape[1],
            scale=q.shape[-1] ** -0.5,
            input_layout="BSND",
            block_size=ctx.page_size,
            sparse_mode=0,
        )

        # 9. FIA returns ``(attention_out, softmax_lse)``; softmax_lse is
        # empty in inference. Reshape back to the caller's flat layout.
        attention_out = result[0]
        return attention_out.view(q.shape)
