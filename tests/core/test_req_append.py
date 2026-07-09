from __future__ import annotations

import pytest
import torch
from minisgl.core import Req, SamplingParams


def _make_req(prompt_len: int = 8, output_len: int = 16, dtype=torch.int32) -> Req:
    return Req(
        input_ids=torch.arange(prompt_len, dtype=dtype),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_append_host_matches_cat_reference():
    req = _make_req(prompt_len=8, output_len=16)
    reference = req.input_ids.clone()

    for token in range(100, 116):
        next_token = torch.tensor([token], dtype=torch.int32)
        reference = torch.cat([reference, next_token])
        req.complete_one()
        req.append_host(next_token)
        assert req.input_ids.is_cpu
        assert torch.equal(req.input_ids, reference)
        assert len(req.input_ids) == req.device_len


def test_append_host_preserves_dtype():
    for dtype in (torch.int32, torch.int64):
        req = _make_req(dtype=dtype)
        req.complete_one()
        req.append_host(torch.tensor([42], dtype=torch.int32))
        assert req.input_ids.dtype == dtype


def test_append_host_never_mutates_written_positions():
    # The radix cache stores slices of `input_ids` by reference (see
    # RadixTreeNode.set_key_value), so previously-appended positions must
    # never change under later appends.
    req = _make_req(prompt_len=4, output_len=8)
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))
    snapshot_view = req.input_ids[:5]  # what a radix key would hold
    snapshot_copy = snapshot_view.clone()

    for token in range(101, 108):
        req.complete_one()
        req.append_host(torch.tensor([token], dtype=torch.int32))

    assert torch.equal(snapshot_view, snapshot_copy)


def test_append_host_overflow_raises():
    # Appending beyond max_device_len must fail loudly, not silently drop.
    req = _make_req(prompt_len=2, output_len=1)
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    req.complete_one()
    with pytest.raises(IndexError):
        req.append_host(torch.tensor([8], dtype=torch.int32))


def test_chunked_req_append_still_forbidden():
    from minisgl.scheduler.prefill import ChunkedReq

    req = ChunkedReq(
        input_ids=torch.arange(4, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=4,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore[arg-type]
    )
    with pytest.raises(NotImplementedError):
        req.append_host(torch.tensor([1], dtype=torch.int32))
