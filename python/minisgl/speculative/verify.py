"""Pure greedy verification for vanilla speculative decoding.

Position alignment between the draft's K proposals and the target's K+1
verify-pass logit rows is the single most bug-prone piece of the system.
Isolating the math here, with no model or cache dependencies, makes the
indexing exhaustively testable without ever loading a model.
"""
from __future__ import annotations

import torch


def verify_drafts(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Greedy-verify K drafted tokens against the target's verify-pass logits.

    The target's verify pass consumes the K+1 inputs ``[t0, d1, ..., dK]``,
    where ``t0`` is the previously-committed last token and ``d1..dK`` are the
    draft proposals. Row ``j`` of ``target_logits`` is the target's prediction
    for the token following input position ``j``.

    Greedy rule: accept ``d_{j+1}`` iff ``argmax(target_logits[j]) == d_{j+1}``.
    Stop at the first mismatch; the recovery token is the target's argmax at
    that position. If all K drafts are accepted, the bonus token at position K
    is appended.

    Args:
        draft_token_ids: 1D int tensor of shape ``(K,)``.
        target_logits:   2D float tensor of shape ``(K+1, vocab)``.

    Returns:
        accepted_tokens: 1D int tensor of length ``accepted_drafts + 1``
            (always at least 1 token).
        accepted_drafts: integer in ``[0, K]``.
    """
    if draft_token_ids.ndim != 1:
        raise ValueError(
            f"draft_token_ids must be 1D, got shape {tuple(draft_token_ids.shape)}"
        )
    if target_logits.ndim != 2:
        raise ValueError(
            f"target_logits must be 2D, got shape {tuple(target_logits.shape)}"
        )
    K = draft_token_ids.shape[0]
    if target_logits.shape[0] != K + 1:
        raise ValueError(
            f"target_logits must have K+1={K + 1} rows, got {target_logits.shape[0]}"
        )

    target_argmax = target_logits.argmax(dim=-1).to(draft_token_ids.dtype)
    matches = target_argmax[:K] == draft_token_ids

    if matches.all():
        accepted_drafts = K
    else:
        # argmin on a 0/1 tensor returns the index of the first 0
        accepted_drafts = int(matches.to(torch.int).argmin().item())

    recovery = target_argmax[accepted_drafts : accepted_drafts + 1]
    accepted_tokens = torch.cat([draft_token_ids[:accepted_drafts], recovery])
    return accepted_tokens, accepted_drafts
