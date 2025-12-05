from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.config.context import Req


@dataclass
class BaseCaptureData:
    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    out_loc: torch.Tensor


def make_out_loc(page_table: torch.Tensor, reqs: List[Req]) -> torch.Tensor:
    from minisgl.kernel_v2 import make_2d_indices

    return make_2d_indices(
        table_2d=page_table,
        ranges=[(req.table_idx, req.cached_len, req.device_len) for req in reqs],
        load_table=True,
    )


def make_positions(device: torch.device, reqs: List[Req]) -> torch.Tensor:
    import torch
    from minisgl.kernel_v2 import make_2d_indices

    return make_2d_indices(
        table_2d=torch.empty((0, 0), dtype=torch.int32, device=device),
        ranges=[(0, req.cached_len, req.device_len) for req in reqs],
        load_table=False,
    )
