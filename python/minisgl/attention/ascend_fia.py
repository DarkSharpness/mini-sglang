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
        # Preserved verbatim from Gate 1.8a — the FIA op call lands in a
        # separate Gate on top of Gate 1.8c's metadata builder.
        raise NotImplementedError(
            "Ascend FIA forward is not implemented until Gate 1.8b"
        )
