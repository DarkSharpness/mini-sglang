from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CaptureData:
    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    out_loc: torch.Tensor
