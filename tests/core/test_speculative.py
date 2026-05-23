"""Vanilla speculative-decoding smoke test.

Exercises ``verify_drafts`` — the pure position-alignment math at the heart of
greedy speculative verification — on hand-crafted logits, no model load
required. Covers all-accept (bonus token), all-reject (recovery token), and
partial-accept at each interior position.

End-to-end engine-level correctness lands with Phase 3, when the real
``VanillaDraft`` is added and HF ``assisted_generation`` becomes available as
the byte-equal reference oracle. Until then, the engine path is exercised by
the local-only test under ``tests/.local/``.
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
    drafts_ids: list[int],
    picks: list[int],
    expected_accepted: int,
    expected_tokens: list[int],
) -> None:
    drafts = torch.tensor(drafts_ids, dtype=torch.int32)
    target_logits = _logits_picking(picks)
    accepted_tokens, accepted_drafts = verify_drafts(drafts, target_logits)
    assert accepted_drafts == expected_accepted, (
        f"{label}: accepted_drafts {accepted_drafts} != {expected_accepted}"
    )
    assert accepted_tokens.tolist() == expected_tokens, (
        f"{label}: tokens {accepted_tokens.tolist()} != {expected_tokens}"
    )


@call_if_main(__name__)
def main():
    # Scenario table: (label, drafts, target_picks, expected_accepted, expected_tokens).
    # Together these cover the K=4 all-accept (bonus token), all-reject (recovery
    # token only), and every interior partial-accept position.
    scenarios = [
        ("all_accept",          [10, 20, 30, 40], [10, 20, 30, 40, 99],  4, [10, 20, 30, 40, 99]),
        ("all_reject",          [10, 20, 30, 40], [5, 99, 99, 99, 99],   0, [5]),
        ("partial_j1",          [10, 20, 30, 40], [10, 777, 99, 99, 99], 1, [10, 777]),
        ("partial_j2",          [10, 20, 30, 40], [10, 20, 777, 99, 99], 2, [10, 20, 777]),
        ("last_draft_rejected", [10, 20, 30, 40], [10, 20, 30, 777, 99], 3, [10, 20, 30, 777]),
    ]

    for label, drafts, picks, expected_accepted, expected_tokens in scenarios:
        _check(label, drafts, picks, expected_accepted, expected_tokens)
        logger.info(f"  {label:24s}  ok  (accepted={expected_accepted}, emitted={expected_tokens})")

    # K=1 boundary: a single rejected draft must still emit the recovery token.
    _check("K1_reject", [42], [5, 99], 0, [5])
    logger.info(f"  {'K1_reject':24s}  ok  (accepted=0, emitted=[5])")

    logger.info("verify_drafts smoke checks passed")
