"""Ascend Fused Infer Attention (FIA) backend.

Gate 2.2c generalises the wrapper from single-request BSND to equal-length
multi-request BSND. The underlying FIA operator was proven to accept
``B >= 1`` batched prefill/decode by the Gate 2.2b contract probe; only the
wrapper's ``len(reqs) != 1`` guard needed lifting.

Supported today:
  * ``B >= 1`` where every real request in ``batch.padded_reqs`` shares the
    same ``extend_len``, ``cached_len`` and ``device_len`` (equal-length
    batch, prefill OR decode).

Explicitly refused (``NotImplementedError``):
  * Ragged batches — any variance across ``extend_len``/``cached_len``/
    ``device_len``. The TND path lands in a later Gate.

The module stays torch-free at import time: ``torch`` and ``minisgl.core``
are pulled in lazily inside method bodies so importing
``minisgl.attention.ascend_fia`` on a CUDA / CPU host is safe. The module
must never reference ``torch_npu`` outside :meth:`AscendFIABackend.forward`.
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
    """Metadata for the equal-length multi-request BSND Ascend FIA path.

    Fields:
        block_table:            ``[B, num_blocks]`` int32 on the KV-cache
                                device. Row ``b`` lists the physical page
                                ids backing request ``b`` — stride-then-
                                divide of the global ``page_table``.
        actual_seq_lengths:     per-request query length; a Python list of
                                length ``B`` with a shared value
                                (equal-length batching). Passed straight to
                                ``torch_npu.npu_fused_infer_attention_score``
                                without a device round-trip.
        actual_seq_lengths_kv:  per-request total KV length (cached + extend);
                                same layout, shared value.
        input_layout:           ``"BSND"``.
        batch_size:             ``B`` — cached so :meth:`forward` can
                                ``reshape`` the flat query without a
                                ``len(...)`` call on the metadata list.
        query_seq_len:          shared ``extend_len`` across the batch —
                                equal to ``actual_seq_lengths[0]``.
        kv_seq_len:             shared ``device_len`` across the batch —
                                equal to ``actual_seq_lengths_kv[0]``.
    """

    block_table: "torch.Tensor"
    actual_seq_lengths: List[int]
    actual_seq_lengths_kv: List[int]
    input_layout: str
    batch_size: int
    query_seq_len: int
    kv_seq_len: int

    def get_last_indices(self, bs: int) -> "torch.Tensor":
        """Return the index of the last query token per request in the flat
        query buffer.

        Under equal-length batching the flat query is laid out as
        ``[req0_tok0, ..., req0_tokS-1, req1_tok0, ...]`` so the last index
        of request ``b`` is ``(b + 1) * query_seq_len - 1``. This matches the
        semantics of ``cu_seqlens_q[1:1+bs] - 1`` used by the CUDA backends.
        """
        # Lazy import so the module stays torch-free at import time.
        import torch

        return (
            torch.arange(
                1,
                bs + 1,
                dtype=torch.int32,
                device=self.block_table.device,
            )
            * self.query_seq_len
            - 1
        )


class AscendFIABackend(BaseAttnBackend):
    """Ascend FIA paged-KV attention backend (equal-length multi-request).

    The constructor signature mirrors what :func:`create_attention_backend`
    already passes to :class:`FlashInferBackend` / :class:`FlashAttentionBackend`
    / :class:`TensorRTLLMBackend`: a single ``ModelConfig`` positional.
    """

    def __init__(self, config: "ModelConfig") -> None:
        self.config = config

    # --------------------------------------------------------------- graph
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # NPU-graph capture wiring lands with a later Gate.
        return None

    def prepare_for_capture(self, batch: "Batch") -> None:
        return None

    def prepare_for_replay(self, batch: "Batch") -> None:
        return None

    # ------------------------------------------------------------ metadata
    def prepare_metadata(self, batch: "Batch") -> None:
        """Build :class:`FIAMetadata` for the equal-length BSND path.

        Accepts ``B >= 1`` when every real request in ``batch.padded_reqs``
        shares the same ``extend_len``, ``cached_len`` and ``device_len``.
        Ragged batches raise :class:`NotImplementedError` — the TND path is a
        future Gate.
        """
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("Ascend FIA metadata received an empty request list")

        # Lazy imports keep the module import-safe on CUDA / CPU hosts and
        # avoid pulling ``minisgl.core`` (which imports torch) at registration
        # time.
        import torch
        from minisgl.core import get_global_ctx

        # Equal-length invariant: every real request in the batch must share
        # the same (extend_len, device_len, cached_len). We derive the shared
        # value from reqs[0] and check the rest — anything ragged is refused
        # with an explicit NotImplementedError instead of silently building a
        # wrong batched metadata.
        head = reqs[0]
        query_seq_len = head.extend_len
        kv_seq_len = head.device_len
        cached_len_head = kv_seq_len - query_seq_len
        for idx, r in enumerate(reqs[1:], start=1):
            r_cached_len = r.device_len - r.extend_len
            if (
                r.extend_len != query_seq_len
                or r.device_len != kv_seq_len
                or r_cached_len != cached_len_head
            ):
                raise NotImplementedError(
                    "Ascend FIA metadata requires all requests in the batch to "
                    "share the same extend_len, cached_len and device_len "
                    "(ragged batches are not supported yet). "
                    f"req[0]: extend_len={query_seq_len} device_len={kv_seq_len} "
                    f"cached_len={cached_len_head}; "
                    f"req[{idx}]: extend_len={r.extend_len} "
                    f"device_len={r.device_len} cached_len={r_cached_len}"
                )

        ctx = get_global_ctx()
        page_table = ctx.page_table
        page_size = ctx.page_size
        num_blocks = _ceil_div(kv_seq_len, page_size)
        batch_size = len(reqs)

        # Stride-then-divide (see fa.py:93 / trtllm.py:117 / Gate 1.8c). Under
        # BnNBsD the global page_table stores raw slots (page_id * page_size +
        # offset); striding by ``page_size`` picks the first slot of each page
        # and dividing recovers the physical page id.
        #
        # For B>=1 we build the block_table row-by-row via a stack — cheap for
        # the batch sizes we care about (max_running_req is O(8)) and keeps
        # the raw-slot → page-id conversion unambiguous under fragmentation.
        row_views = [
            page_table[r.table_idx, : num_blocks * page_size : page_size]
            for r in reqs
        ]
        block_table = (torch.stack(row_views, dim=0) // page_size).contiguous()

        # Structural invariants FIA relies on. Cheap enough to keep at runtime.
        assert block_table.shape == (batch_size, num_blocks), (
            f"expected block_table shape ({batch_size}, {num_blocks}), "
            f"got {tuple(block_table.shape)}"
        )
        assert block_table.dtype == torch.int32, (
            f"expected block_table dtype int32, got {block_table.dtype}"
        )
        assert block_table.device == page_table.device, (
            f"expected block_table on {page_table.device}, got {block_table.device}"
        )

        batch.attn_metadata = FIAMetadata(
            block_table=block_table,
            actual_seq_lengths=[query_seq_len] * batch_size,
            actual_seq_lengths_kv=[kv_seq_len] * batch_size,
            input_layout="BSND",
            batch_size=batch_size,
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
        """Run one paged-KV FIA attention step (equal-length BSND).

        Execution order:

        1. validate ``batch.attn_metadata`` is a :class:`FIAMetadata`;
        2. validate the flat query token count matches
           ``batch_size * query_seq_len`` (guards a caller that mutated
           ``padded_reqs`` between :meth:`prepare_metadata` and here);
        3. store this layer's new K/V into the paged cache — the scatter is
           keyed by ``batch.out_loc`` (per-request raw slots concatenated by
           the scheduler), so page isolation is inherited from the caller;
        4. reshape the flat query ``[B*S, Hq, D]`` to BSND ``[B, S, Hq, D]``;
        5. fetch the FIA-native BnNBsD cache tensors — passed verbatim;
        6. build a shared causal ``atten_mask`` ``[S, padded_kv_len]``
           (or ``None`` for the single-Q decode path); mask visibility
           includes the shared cached prefix;
        7. dynamic-import ``torch_npu`` and call
           :func:`torch_npu.npu_fused_infer_attention_score`;
        8. take the first tensor of the returned tuple (softmax_lse is
           unused in inference mode);
        9. view back to the flat ``[B*S, Hq, D]`` shape the caller expects.
        """
        # 1. metadata type check
        metadata = batch.attn_metadata
        if not isinstance(metadata, FIAMetadata):
            raise TypeError(
                "Ascend FIA forward expects batch.attn_metadata to be "
                f"FIAMetadata, got {type(metadata).__name__}"
            )

        batch_size = metadata.batch_size
        query_seq_len = metadata.query_seq_len
        expected_tokens = batch_size * query_seq_len

        # 2. flat token count must match the metadata layout — catches a caller
        # that mutated ``padded_reqs`` (or ``batch.out_loc``) between
        # ``prepare_metadata`` and here.
        if q.shape[0] != expected_tokens:
            raise ValueError(
                "Ascend FIA forward: flat query token count "
                f"{q.shape[0]} does not match batch_size * query_seq_len "
                f"({batch_size} * {query_seq_len} = {expected_tokens}); "
                "either the metadata or the flat query was mutated after "
                "prepare_metadata()"
            )

        # Lazy imports keep the module import-safe on CUDA / CPU hosts.
        import torch
        from minisgl.core import get_global_ctx

        ctx = get_global_ctx()

        # 3. Persist this layer's new K/V into the paged cache. ``out_loc``
        # already carries the per-request raw slots concatenated in flat order
        # by the scheduler; store_kv scatters slot-by-slot, so page isolation
        # between requests is preserved without any per-request looping here.
        ctx.kv_cache.store_kv(k, v, batch.out_loc, layer_id)

        # 4. Reshape flat [B*S, Hq, D] -> [B, S, Hq, D]. ``reshape`` handles
        # both contiguous and non-contiguous q (falling back to a copy only if
        # strictly necessary); the model's Attention.forward already produces
        # a contiguous flat q, so this is a view in practice.
        head_dim = q.shape[-1]
        num_qo_heads = q.shape[-2]
        query_bsnd = q.reshape(batch_size, query_seq_len, num_qo_heads, head_dim)

        # 5. BnNBsD paged caches — passed verbatim, no permute / contiguous.
        key_cache = ctx.kv_cache.k_cache(layer_id)
        value_cache = ctx.kv_cache.v_cache(layer_id)

        # 6. Causal mask.
        #  * decode (query_seq_len == 1): FIA elides masking with atten_mask=None.
        #  * prefill: shared [S, padded_kv_len] causal mask offset by the
        #    common cached prefix. FIA requires the KV axis to be padded to
        #    ``num_blocks * page_size`` (Gate 1.8d); padded columns are all
        #    True (== masked out) so they cannot contribute to any row. The
        #    same mask is broadcast across all B requests — legal because
        #    every real request in the batch shares the same cached prefix
        #    and the same extend length under the equal-length invariant.
        if query_seq_len == 1:
            atten_mask = None
        else:
            padded_kv_len = metadata.block_table.shape[1] * ctx.page_size
            cached_len = metadata.kv_seq_len - query_seq_len
            q_pos = cached_len + torch.arange(query_seq_len, device=q.device)
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
        # binding. ``num_heads`` is Hq from the flat query (q.shape[-2]);
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
            num_heads=num_qo_heads,
            num_key_value_heads=key_cache.shape[1],
            scale=head_dim ** -0.5,
            input_layout="BSND",
            block_size=ctx.page_size,
            sparse_mode=0,
        )

        # 9. FIA returns ``(attention_out, softmax_lse)``; softmax_lse is
        # empty in inference. Reshape back to the caller's flat layout —
        # attention_out is [B, S, Hq, D] contiguous, so ``.view`` is a
        # zero-copy view of the same storage.
        attention_out = result[0]
        return attention_out.view(q.shape)
