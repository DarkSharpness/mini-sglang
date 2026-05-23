"""Vanilla speculative-decoding smoke test.

Exercises ``verify_drafts`` — the pure position-alignment math at the heart of
greedy speculative verification — on hand-crafted logits, no model load
required. Covers all-accept (bonus token), all-reject (recovery token), and
partial-accept at each interior position.

End-to-end engine-level correctness (real ``StandaloneDraft`` against an HF
``assisted_generation`` byte-equal oracle) is exercised by the local-only test
under ``tests/.local/``, which needs GPU + model weights not available in CI.
"""
from __future__ import annotations

import torch

from minisgl.speculative.verify import verify_drafts
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _logits_picking(picks: list[int], vocab: int = 1024) -> torch.Tensor:
    """Build a (T, vocab) tensor whose argmax at row j is picks[j]."""
    logits = torch.full((len(picks), vocab), -1e9, dtype=torch.float32)
    for j, tok in enumerate(picks):
        logits[j, tok] = 1.0
    return logits


def _check(
    label: str,
    draft_ids: list[int],
    picks: list[int],
    expected_correct: int,
    expected_accept: list[int],
) -> None:
    draft_tokens = torch.tensor(draft_ids, dtype=torch.int32)
    target_logits = _logits_picking(picks)
    accept_tokens, num_correct_drafts = verify_drafts(draft_tokens, target_logits)
    assert num_correct_drafts == expected_correct, (
        f"{label}: num_correct_drafts {num_correct_drafts} != {expected_correct}"
    )
    assert accept_tokens.tolist() == expected_accept, (
        f"{label}: accept_tokens {accept_tokens.tolist()} != {expected_accept}"
    )


@call_if_main(__name__)
def main():
    # Scenario table: (label, draft_tokens, target_picks, expected_correct, expected_accept).
    # Together these cover the K=4 all-correct (with bonus token), all-reject
    # (bonus token only), and every interior partial-accept position.
    scenarios = [
        ("all_correct",         [10, 20, 30, 40], [10, 20, 30, 40, 99],  4, [10, 20, 30, 40, 99]),
        ("all_reject",          [10, 20, 30, 40], [5, 99, 99, 99, 99],   0, [5]),
        ("partial_j1",          [10, 20, 30, 40], [10, 777, 99, 99, 99], 1, [10, 777]),
        ("partial_j2",          [10, 20, 30, 40], [10, 20, 777, 99, 99], 2, [10, 20, 777]),
        ("last_draft_rejected", [10, 20, 30, 40], [10, 20, 30, 777, 99], 3, [10, 20, 30, 777]),
    ]

    for label, draft_ids, picks, expected_correct, expected_accept in scenarios:
        _check(label, draft_ids, picks, expected_correct, expected_accept)
        logger.info(f"  {label:24s}  ok  (correct={expected_correct}, accept={expected_accept})")

    # K=1 boundary: a single rejected draft must still emit the bonus token.
    _check("K1_reject", [42], [5, 99], 0, [5])
    logger.info(f"  {'K1_reject':24s}  ok  (correct=0, accept=[5])")

    logger.info("verify_drafts smoke checks passed")
