from __future__ import annotations

import torch


def verify_drafts(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """
    Greedy-verify K drafted tokens against the target's K+1 verify-pass logits.
    Accept each draft while it matches the target's argmax; stop at the first
    mismatch and take the target's argmax there as the bonus token.

    Args:
        draft_tokens (torch.Tensor): Draft proposals. Shape: (K,).
        target_logits (torch.Tensor): Target verify-pass logits. Shape: (K+1, vocab).
    Returns:
        accept_tokens (torch.Tensor): Accepted tokens incl. bonus. Shape: (num_correct_drafts + 1,).
        num_correct_drafts (int): Drafts accepted, excludes bonus. In [0, K].
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
