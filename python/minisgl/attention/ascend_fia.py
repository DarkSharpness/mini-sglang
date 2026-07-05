"""Ascend Fused Infer Attention (FIA) backend.

Gate 2.2c generalises the wrapper from single-request BSND to equal-length
multi-request BSND. Gate 2.2f extends the wrapper to **ragged prefill**
where every request has ``cached_len == 0`` but individual ``extend_len``
values differ. The underlying FIA operator was proven to accept ragged
BSND prefill (see Gate 2.2e-r1 contract probe) via right-padded query
plus per-batch ``actual_seq_lengths``.

Supported today:
  * ``B >= 1`` decode (all requests ``extend_len == 1``, arbitrary
    ``cached_len`` shared across the batch).
  * ``B >= 1`` prefill when every real request shares the same
    ``extend_len``/``cached_len``/``device_len`` (equal-length path).
  * ``B >= 1`` prefill when every real request has ``cached_len == 0``
    while ``extend_len`` (== ``device_len``) may differ per request
    (ragged path, Gate 2.2f).

Explicitly refused (``NotImplementedError``):
  * Ragged batches where any request has ``cached_len != 0`` — the mixed
    cached-prefix ragged case still lands in a later Gate.
  * Decode batches where ``cached_len`` differs per request — the
    per-batch KV lengths are already supported by the operator but the
    scheduler currently never emits this shape.

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
    """Metadata for the BSND Ascend FIA path.

    Fields:
        block_table:            ``[B, max_blocks]`` int32 on the KV-cache
                                device. Row ``b`` lists the physical page
                                ids backing request ``b`` — stride-then-
                                divide of the global ``page_table``. Under
                                ragged prefill every row is padded to the
                                same ``max_blocks`` width; unused columns
                                are filled with ``0`` (an in-range but
                                unused page id — safe because the causal
                                mask hides every slot in a padded page for
                                the padded query rows).
        actual_seq_lengths:     per-request query length (a Python list of
                                length ``B``). Under equal-length batching
                                every entry is the same; under ragged
                                prefill entries differ per request. Passed
                                straight to
                                ``torch_npu.npu_fused_infer_attention_score``
                                without a device round-trip.
        actual_seq_lengths_kv:  per-request total KV length (cached +
                                extend); same shape as
                                ``actual_seq_lengths``.
        input_layout:           ``"BSND"``.
        batch_size:             ``B`` — cached so :meth:`forward` can build
                                the packed BSND query without a ``len(...)``
                                call on the metadata list.
        query_seq_lens:         per-request query length as a plain Python
                                ``list[int]``. Duplicates
                                ``actual_seq_lengths`` but kept separate to
                                keep the forward path free of aliasing —
                                the operator kwarg may be mutated by a
                                future guard without affecting the pack
                                logic.
        kv_seq_lens:            per-request total KV length as ``list[int]``.
        max_query_len:          ``max(query_seq_lens)`` — the padded S
                                dimension of the BSND query tensor handed
                                to FIA.
        query_offsets:          cumulative start offsets of each request
                                in the flat query buffer;
                                ``[0, q0, q0+q1, ..., sum_qi]`` of length
                                ``B + 1``. Used by :meth:`forward` to scatter
                                the flat query into the padded BSND tensor
                                and gather the output back.
        query_seq_len:          shared ``extend_len`` across the batch under
                                the equal-length path. Under ragged prefill
                                this is ``None`` — callers must consult
                                ``query_seq_lens`` / ``max_query_len``
                                instead.
        kv_seq_len:             shared ``device_len`` across the batch under
                                the equal-length path. ``None`` under
                                ragged prefill.
    """

    block_table: "torch.Tensor"
    actual_seq_lengths: List[int]
    actual_seq_lengths_kv: List[int]
    input_layout: str
    batch_size: int
    query_seq_lens: List[int]
    kv_seq_lens: List[int]
    max_query_len: int
    query_offsets: List[int]
    # Legacy equal-length shortcuts (None under ragged prefill).
    query_seq_len: "int | None"
    kv_seq_len: "int | None"

    def get_last_indices(self, bs: int) -> "torch.Tensor":
        """Return the index of the last query token per request in the flat
        query buffer.

        The flat query is laid out as
        ``[req0_tok0, ..., req0_tokQ0-1, req1_tok0, ..., req1_tokQ1-1, ...]``
        so the last index of request ``b`` is ``query_offsets[b+1] - 1``.
        Under equal-length batching this reduces to
        ``(b + 1) * query_seq_len - 1`` — matching the semantics of
        ``cu_seqlens_q[1:1+bs] - 1`` used by the CUDA backends and
        preserving the Gate 2.2c contract.
        """
        # Lazy import so the module stays torch-free at import time.
        import torch

        # ``query_offsets`` has length B+1; last index of request b is
        # offsets[b+1] - 1. Slice offsets[1:1+bs] gives one value per request.
        ends = self.query_offsets[1 : 1 + bs]
        return torch.tensor(
            [e - 1 for e in ends],
            dtype=torch.int32,
            device=self.block_table.device,
        )


class AscendFIABackend(BaseAttnBackend):
    """Ascend FIA paged-KV attention backend (equal-length + ragged prefill).

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
        """Build :class:`FIAMetadata` for the BSND path.

        Accepts:
          * equal-length batches (``B >= 1``, shared
            ``extend_len``/``cached_len``/``device_len``) — legacy path
            from Gate 2.2c; and
          * ragged prefill batches (``B >= 1``) where **every** request
            has ``cached_len == 0`` while ``extend_len`` may vary. Ragged
            decode batches (different ``cached_len``) still raise
            :class:`NotImplementedError`, as do ragged prefill batches
            with any non-zero ``cached_len``.
        """
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("Ascend FIA metadata received an empty request list")

        # Lazy imports keep the module import-safe on CUDA / CPU hosts and
        # avoid pulling ``minisgl.core`` (which imports torch) at registration
        # time.
        import torch
        from minisgl.core import get_global_ctx

        # Classify the batch:
        #   * equal-length (all reqs share (extend_len, cached_len, device_len))
        #     — the legacy path from Gate 2.2c.
        #   * ragged prefill (all cached_len == 0, extend_len may vary)
        #     — the Gate 2.2f path.
        head = reqs[0]
        head_query = head.extend_len
        head_kv = head.device_len
        head_cached = head_kv - head_query

        all_equal = True
        all_cached_zero = head_cached == 0
        pure_decode = head_query == 1
        for r in reqs[1:]:
            r_cached = r.device_len - r.extend_len
            if (
                r.extend_len != head_query
                or r.device_len != head_kv
                or r_cached != head_cached
            ):
                all_equal = False
            if r_cached != 0:
                all_cached_zero = False
            if r.extend_len != 1:
                pure_decode = False

        if not (all_equal or all_cached_zero or pure_decode):
            # Ragged batch that is neither purely ragged prefill (all
            # cached_len == 0) nor purely decode (all extend_len == 1) and
            # has some non-zero cached_len on at least one request. This is
            # the mixed-prefix-hit ragged case with extend_len > 1 —
            # reserved for a later Gate. Report the earliest offender so
            # the caller can trace the batch.
            for idx, r in enumerate(reqs):
                r_cached = r.device_len - r.extend_len
                if r_cached != 0 and r.extend_len > 1:
                    raise NotImplementedError(
                        "Ascend FIA metadata: ragged batches with a non-zero "
                        "cached_len and extend_len>1 are not supported yet "
                        "(Gate 2.2f covers the cached_len==0 ragged prefill "
                        "case and the pure-decode ragged case). Offender "
                        f"req[{idx}]: extend_len={r.extend_len} "
                        f"device_len={r.device_len} cached_len={r_cached}."
                    )
            # Fallback — shouldn't reach here given the classification above.
            raise NotImplementedError(
                "Ascend FIA metadata: unsupported ragged batch shape."
            )

        ctx = get_global_ctx()
        page_table = ctx.page_table
        page_size = ctx.page_size
        batch_size = len(reqs)

        # Per-request block-count and shared max across the batch. Under the
        # equal-length path every req has the same ``kv_seq_len`` and hence
        # the same block count; under ragged prefill each row's block count
        # is ``ceil(extend_len / page_size)``.
        blocks_per_req = [_ceil_div(r.device_len, page_size) for r in reqs]
        max_blocks = max(blocks_per_req)

        # Stride-then-divide (see fa.py:93 / trtllm.py:117 / Gate 1.8c). Under
        # BnNBsD the global page_table stores raw slots (page_id * page_size +
        # offset); striding by ``page_size`` picks the first slot of each page
        # and dividing recovers the physical page id.
        #
        # We build ``block_table`` row-by-row: each row is the request's own
        # page-id sequence, right-padded with ``0`` to ``max_blocks`` under
        # ragged prefill. Padding page ``0`` is safe because the causal mask
        # hides every KV slot that is not in the real KV region for that
        # request; the operator therefore never reads through the padded
        # entries. (The scheduler always keeps real page id ``0`` allocated
        # to some other request; picking any in-range value works.)
        rows: List[torch.Tensor] = []
        for r, nb in zip(reqs, blocks_per_req):
            real_row = page_table[r.table_idx, : nb * page_size : page_size] // page_size
            if nb < max_blocks:
                pad = torch.zeros(max_blocks - nb, dtype=real_row.dtype, device=real_row.device)
                rows.append(torch.cat([real_row, pad], dim=0))
            else:
                rows.append(real_row)
        block_table = torch.stack(rows, dim=0).contiguous()

        assert block_table.shape == (batch_size, max_blocks), (
            f"expected block_table shape ({batch_size}, {max_blocks}), "
            f"got {tuple(block_table.shape)}"
        )
        assert block_table.dtype == torch.int32, (
            f"expected block_table dtype int32, got {block_table.dtype}"
        )
        assert block_table.device == page_table.device, (
            f"expected block_table on {page_table.device}, got {block_table.device}"
        )

        query_seq_lens = [r.extend_len for r in reqs]
        kv_seq_lens = [r.device_len for r in reqs]
        max_query_len = max(query_seq_lens)

        # Cumulative offsets: [0, q0, q0+q1, ..., sum_qi]. Length B+1.
        query_offsets = [0]
        acc = 0
        for q in query_seq_lens:
            acc += q
            query_offsets.append(acc)

        batch.attn_metadata = FIAMetadata(
            block_table=block_table,
            actual_seq_lengths=list(query_seq_lens),
            actual_seq_lengths_kv=list(kv_seq_lens),
            input_layout="BSND",
            batch_size=batch_size,
            query_seq_lens=list(query_seq_lens),
            kv_seq_lens=list(kv_seq_lens),
            max_query_len=max_query_len,
            query_offsets=query_offsets,
            query_seq_len=head_query if all_equal else None,
            kv_seq_len=head_kv if all_equal else None,
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
        """Run one paged-KV FIA attention step (equal-length or ragged BSND).

        Execution order:

        1. validate ``batch.attn_metadata`` is a :class:`FIAMetadata`;
        2. validate the flat query token count matches
           ``sum(query_seq_lens)`` (guards a caller that mutated
           ``padded_reqs`` between :meth:`prepare_metadata` and here);
        3. store this layer's new K/V into the paged cache — the scatter is
           keyed by ``batch.out_loc`` (per-request raw slots concatenated by
           the scheduler in the same order as ``query_offsets``), so page
           isolation is inherited from the caller;
        4. pack the flat query ``[sum_q, Hq, D]`` into padded BSND
           ``[B, max_query_len, Hq, D]`` — real rows first, tail rows zero;
        5. fetch the FIA-native BnNBsD cache tensors — passed verbatim;
        6. build the atten_mask:
           * decode (max_query_len == 1): ``None``;
           * equal-length prefill: shared ``[max_query_len, padded_kv_len]``
             causal mask offset by the common cached prefix;
           * ragged prefill: per-batch ``[B, 1, max_query_len, padded_kv_len]``
             causal mask; padded query rows are fully masked;
        7. dynamic-import ``torch_npu`` and call
           :func:`torch_npu.npu_fused_infer_attention_score`;
        8. take the first tensor of the returned tuple (softmax_lse is
           unused in inference mode);
        9. unpad the ``[B, max_query_len, Hq, D]`` output back to the flat
           ``[sum_q, Hq, D]`` shape the caller expects, then reshape to the
           original ``q`` layout (``[sum_q, Hq*D]``).
        """
        # 1. metadata type check
        metadata = batch.attn_metadata
        if not isinstance(metadata, FIAMetadata):
            raise TypeError(
                "Ascend FIA forward expects batch.attn_metadata to be "
                f"FIAMetadata, got {type(metadata).__name__}"
            )

        batch_size = metadata.batch_size
        query_seq_lens = metadata.query_seq_lens
        max_query_len = metadata.max_query_len
        query_offsets = metadata.query_offsets
        expected_tokens = query_offsets[-1]

        # 2. flat token count must match the metadata layout — catches a caller
        # that mutated ``padded_reqs`` (or ``batch.out_loc``) between
        # ``prepare_metadata`` and here.
        if q.shape[0] != expected_tokens:
            raise ValueError(
                "Ascend FIA forward: flat query token count "
                f"{q.shape[0]} does not match sum(query_seq_lens) "
                f"({query_seq_lens} -> {expected_tokens}); "
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

        # 4. Pack flat [sum_q, Hq, D] -> padded BSND [B, max_query_len, Hq, D].
        head_dim = q.shape[-1]
        num_qo_heads = q.shape[-2]
        if all(q_len == max_query_len for q_len in query_seq_lens):
            # Uniform S — a single reshape is a view over the flat q buffer.
            query_bsnd = q.reshape(batch_size, max_query_len, num_qo_heads, head_dim)
        else:
            # Ragged: allocate a zero-init padded tensor and copy each
            # request's rows into place. Tail rows stay zero — the mask
            # ensures they never contribute to any output row (and the
            # operator returns zero for fully-masked rows, per Gate 2.2e-r1).
            query_bsnd = q.new_zeros((batch_size, max_query_len, num_qo_heads, head_dim))
            for b, (start, end) in enumerate(zip(query_offsets[:-1], query_offsets[1:])):
                length = end - start
                query_bsnd[b, :length] = q[start:end]

        # 5. BnNBsD paged caches — passed verbatim, no permute / contiguous.
        key_cache = ctx.kv_cache.k_cache(layer_id)
        value_cache = ctx.kv_cache.v_cache(layer_id)

        # 6. Causal mask.
        #  * decode (max_query_len == 1): FIA elides masking with atten_mask=None.
        #  * equal-length prefill: shared [S, padded_kv_len] causal mask offset
        #    by the common cached prefix. Same mask broadcast across B.
        #  * ragged prefill: per-batch [B, 1, S, padded_kv_len] causal mask.
        #    Padded query rows (index >= this request's query_seq_len) are
        #    fully masked (all True) so they cannot contribute to real rows.
        padded_kv_len = metadata.block_table.shape[1] * ctx.page_size
        if max_query_len == 1:
            atten_mask = None
        elif metadata.query_seq_len is not None:
            # Equal-length prefill: shared 2D causal mask (Gate 2.2c).
            cached_len = metadata.kv_seq_len - metadata.query_seq_len
            q_pos = cached_len + torch.arange(metadata.query_seq_len, device=q.device)
            k_pos = torch.arange(padded_kv_len, device=q.device)
            atten_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        else:
            # Ragged prefill (Gate 2.2f): per-batch causal mask.
            atten_mask = torch.ones(
                (batch_size, 1, max_query_len, padded_kv_len),
                dtype=torch.bool,
                device=q.device,
            )
            k_pos_row = torch.arange(padded_kv_len, device=q.device)
            for b, (q_len, kv_len) in enumerate(zip(query_seq_lens, metadata.kv_seq_lens)):
                cached_len_b = kv_len - q_len
                q_pos_b = cached_len_b + torch.arange(q_len, device=q.device)
                # Standard causal for real query rows: mask when k > q.
                atten_mask[b, 0, :q_len, :] = k_pos_row.unsqueeze(0) > q_pos_b.unsqueeze(1)
                # Padded query rows (index q_len..max_query_len-1) stay fully
                # masked (all True) — already the initial value.

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
        # empty in inference. Under the equal-length path attention_out is
        # already [B, S, Hq, D] and ``.view(q.shape)`` is a zero-copy view.
        # Under the ragged path we gather each request's real rows back into
        # the flat layout before reshape.
        attention_out = result[0]
        if all(q_len == max_query_len for q_len in query_seq_lens):
            return attention_out.view(q.shape)
        flat = q.new_empty((expected_tokens, num_qo_heads, head_dim))
        for b, (start, end) in enumerate(zip(query_offsets[:-1], query_offsets[1:])):
            length = end - start
            flat[start:end] = attention_out[b, :length]
        return flat.view(q.shape)
