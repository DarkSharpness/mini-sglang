"""Ascend Fused Infer Attention (FIA) backend — Gate 1.8a skeleton.

This module provides the class scaffolding + attribute wiring for the
BnNBsD paged-KV attention path on 910B1. All runtime behaviour is deferred
to Gate 1.8b:

* graph capture hooks are no-op (real capture wiring lands with FIA).
* ``prepare_metadata`` is a no-op.
* ``forward`` explicitly raises :class:`NotImplementedError`.

The module deliberately avoids importing ``torch_npu`` (and any Ascend
runtime bindings). It must stay import-safe on CUDA / CPU hosts so the
attention registry keeps its lazy semantics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import BaseAttnBackend

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch
    from minisgl.models import ModelConfig


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
        # Real NPU-graph capture wiring lands with the FIA call in Gate 1.8b.
        return None

    def prepare_for_capture(self, batch: "Batch") -> None:
        return None

    def prepare_for_replay(self, batch: "Batch") -> None:
        return None

    # ------------------------------------------------------------ metadata
    def prepare_metadata(self, batch: "Batch") -> None:
        # No FIA metadata layout is committed until Gate 1.8b lands.
        return None

    # ------------------------------------------------------------- forward
    def forward(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        layer_id: int,
        batch: "Batch",
    ) -> "torch.Tensor":
        raise NotImplementedError(
            "Ascend FIA forward is not implemented until Gate 1.8b"
        )
