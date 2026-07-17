from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Sequence, Tuple

import torch


def propose_ngram(
    input_ids: torch.Tensor,
    *,
    min_ngram: int,
    max_ngram: int,
    max_tokens: int,
) -> List[int]:
    """CPU drafter: longest-then-recent suffix match → up to max_tokens continuation.

    Called once per greedy request on the scheduler critical path (overlap is off
    when spec is on, so this cost is fully exposed).
    """
    assert input_ids.is_cpu
    assert 1 <= min_ngram <= max_ngram

    if max_tokens <= 0:
        return []

    total = int(input_ids.numel())

    for ngram_size in range(min(max_ngram, total - 1), min_ngram - 1, -1):
        suffix_start = total - ngram_size

        # windows[i] is input_ids[i : i + ngram_size]; dropping row suffix_start onwards
        # keeps the suffix from matching itself.
        windows = input_ids.unfold(0, ngram_size, 1)[:suffix_start]
        hits = (windows == input_ids[suffix_start:]).all(dim=1)

        if not bool(hits.any()):
            continue

        # Nearby occurrences generally reflect the most relevant local context, so take the
        # latest match: argmax over the flipped mask finds the last hit.
        match_start = suffix_start - 1 - int(hits.flip(0).to(torch.uint8).argmax())

        continuation_start = match_start + ngram_size
        continuation_end = min(continuation_start + max_tokens, total)

        if continuation_start < continuation_end:
            return input_ids[continuation_start:continuation_end].tolist()

    return []


def accept_greedy(
    draft_tokens: Sequence[int], target_tokens: Sequence[int]
) -> Tuple[List[int], int]:
    """Accept the matching draft prefix and one target correction/bonus.

    target_tokens[j] is the model's argmax at query position j (fed draft/committed
    token j). Accept draft d_j iff d_j == target[j-1]; always emit target[num_accepted]
    as the correction or guaranteed bonus. Returns (committed_tokens, num_accepted_drafts).
    """
    assert len(target_tokens) == len(draft_tokens) + 1

    num_accepted = 0

    while (
        num_accepted < len(draft_tokens)
        and draft_tokens[num_accepted] == target_tokens[num_accepted]
    ):
        num_accepted += 1

    # Dry run for drafts [11, 22, 33] and targets [11, 99, 44, 55]:
    # d1 == g0, but d2 != g1. We therefore commit [g0, g1] = [11, 99].
    # g1 is the correction at the first mismatch. If all three drafts matched,
    # target_tokens[3] would instead be the guaranteed bonus token.
    return list(target_tokens[: num_accepted + 1]), num_accepted


class VerifyResolve(NamedTuple):
    committed: List[int]
    num_accepted: int
    finished: bool


def resolve_verify(
    draft_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    remain_len: int,
    eos_token_id: int | None,
    ignore_eos: bool,
) -> VerifyResolve:
    """Accept drafts, then truncate at max_tokens / EOS for commit.

    num_accepted is the verifier's draft-accept count (vLLM/SGLang-style) and is
    not clamped by stop truncation — that only shortens committed/emitted tokens.
    """
    committed, num_accepted = accept_greedy(draft_tokens, target_tokens)
    committed = committed[:remain_len]
    finished = len(committed) == remain_len
    if not ignore_eos and eos_token_id is not None:
        for i, token in enumerate(committed):
            if token == eos_token_id:
                committed = committed[: i + 1]
                finished = True
                break
    assert len(committed) > 0
    return VerifyResolve(committed, num_accepted, finished)


@dataclass
class SpecMetrics:
    """Counters for mean accepted tokens/verify, acceptance rate, draft hit rate."""

    verify_requests: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    no_draft_requests: int = 0

    def record(self, num_draft: int, num_accepted: int, num_emitted: int) -> None:
        self.verify_requests += 1
        self.drafted_tokens += num_draft
        self.accepted_tokens += num_accepted
        self.emitted_tokens += num_emitted
        self.no_draft_requests += num_draft == 0
