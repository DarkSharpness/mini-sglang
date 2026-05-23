"""Pure greedy verification for vanilla speculative decoding.

Position alignment between the draft's K proposals and the target's K+1
verify-pass logit rows is the single most bug-prone piece of the system.
Isolating the math here, with no model or cache dependencies, makes the
indexing exhaustively testable without ever loading a model.
"""
from __future__ import annotations

import torch


def verify_drafts(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Greedy-verify K drafted tokens against the target's verify-pass logits.

    The target's verify pass consumes the K+1 inputs ``[t0, d1, ..., dK]``,
    where ``t0`` is the previously-committed last token and ``d1..dK`` are the
    draft proposals. Row ``j`` of ``target_logits`` is the target's prediction
    for the token following input position ``j``.

    Greedy rule: accept ``d_{j+1}`` iff ``argmax(target_logits[j]) == d_{j+1}``.
    Stop at the first mismatch; the bonus token is the target's argmax at that
    position. If all K drafts are correct, the bonus token at position K is
    appended.

    Args:
        draft_tokens: 1D int tensor of shape ``(K,)``.
        target_logits: 2D float tensor of shape ``(K+1, vocab)``.

    Returns:
        accept_tokens: 1D int tensor of length ``num_correct_drafts + 1``
            (always at least 1 token; includes the bonus token).
        num_correct_drafts: integer in ``[0, K]`` (drafts only, excludes bonus).
    """
    if draft_tokens.ndim != 1:
        raise ValueError(
            f"draft_tokens must be 1D, got shape {tuple(draft_tokens.shape)}"
        )
    if target_logits.ndim != 2:
        raise ValueError(
            f"target_logits must be 2D, got shape {tuple(target_logits.shape)}"
        )
    K = draft_tokens.shape[0]
    if target_logits.shape[0] != K + 1:
        raise ValueError(
            f"target_logits must have K+1={K + 1} rows, got {target_logits.shape[0]}"
        )

    target_argmax = target_logits.argmax(dim=-1).to(draft_tokens.dtype)
    matches = target_argmax[:K] == draft_tokens

    if matches.all():
        num_correct_drafts = K
    else:
        # argmin on a 0/1 tensor returns the index of the first 0
        num_correct_drafts = int(matches.to(torch.int).argmin().item())

    bonus_token = target_argmax[num_correct_drafts : num_correct_drafts + 1]
    accept_tokens = torch.cat([draft_tokens[:num_correct_drafts], bonus_token])
    return accept_tokens, num_correct_drafts
