"""Unit tests for the pure greedy verification function."""
from __future__ import annotations

import pytest
import torch

from minisgl.speculative.verify import verify_drafts


def _logits_picking(picks: list[int], vocab: int = 1024) -> torch.Tensor:
    """Build a (T, vocab) tensor whose argmax at row j is picks[j]."""
    logits = torch.full((len(picks), vocab), -1e9, dtype=torch.float32)
    for j, tok in enumerate(picks):
        logits[j, tok] = 1.0
    return logits


def _draft(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32)


# Scenario table: (label, drafts, target_picks, expected_accepted_drafts, expected_tokens)
# Covers all-accept, all-reject, partial accept at every j, and the j=K-1 boundary.
_SCENARIOS = [
    ("all_accept",          [10, 20, 30, 40], [10, 20, 30, 40, 99],  4, [10, 20, 30, 40, 99]),
    ("all_reject",          [10, 20, 30, 40], [5, 99, 99, 99, 99],   0, [5]),
    ("partial_j0",          [10, 20, 30, 40], [777, 99, 99, 99, 99], 0, [777]),
    ("partial_j1",          [10, 20, 30, 40], [10, 777, 99, 99, 99], 1, [10, 777]),
    ("partial_j2",          [10, 20, 30, 40], [10, 20, 777, 99, 99], 2, [10, 20, 777]),
    ("last_draft_rejected", [10, 20, 30, 40], [10, 20, 30, 777, 99], 3, [10, 20, 30, 777]),
]


@pytest.mark.parametrize(
    "drafts_ids,picks,expected_accepted,expected_tokens",
    [(d, p, a, t) for _, d, p, a, t in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
def test_scenarios(drafts_ids, picks, expected_accepted, expected_tokens):
    drafts = _draft(drafts_ids)
    target_logits = _logits_picking(picks)
    accepted_tokens, accepted_drafts = verify_drafts(drafts, target_logits)
    assert accepted_drafts == expected_accepted
    assert accepted_tokens.tolist() == expected_tokens


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_varying_K_all_accept(K: int):
    drafts = _draft(list(range(100, 100 + K)))
    target_logits = _logits_picking(list(range(100, 100 + K)) + [999])
    accepted_tokens, accepted_drafts = verify_drafts(drafts, target_logits)
    assert accepted_drafts == K
    assert accepted_tokens.tolist() == list(range(100, 100 + K)) + [999]


def test_K_eq_1_reject():
    drafts = _draft([42])
    target_logits = _logits_picking([5, 99])
    accepted_tokens, accepted_drafts = verify_drafts(drafts, target_logits)
    assert accepted_drafts == 0
    assert accepted_tokens.tolist() == [5]


def test_dtype_preserved_int64():
    drafts = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    target_logits = _logits_picking([10, 20, 30, 40, 99])
    accepted_tokens, _ = verify_drafts(drafts, target_logits)
    assert accepted_tokens.dtype == torch.int64


def test_invalid_draft_dim():
    with pytest.raises(ValueError, match="draft_token_ids must be 1D"):
        verify_drafts(torch.tensor([[1, 2]]), torch.zeros((3, 10)))


def test_invalid_logits_dim():
    with pytest.raises(ValueError, match="target_logits must be 2D"):
        verify_drafts(torch.tensor([1, 2]), torch.zeros((3, 4, 10)))


def test_logits_wrong_length():
    with pytest.raises(ValueError, match="K\\+1=4 rows, got 3"):
        verify_drafts(torch.tensor([1, 2, 3]), torch.zeros((3, 10)))
