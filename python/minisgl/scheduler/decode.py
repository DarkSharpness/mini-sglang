from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req

from .spec import SpecMetrics, propose_ngram


@dataclass
class DecodeManager:
    page_size: int
    spec_algorithm: str = "none"
    spec_num_draft: int = 4
    spec_ngram_min: int = 1
    spec_ngram_max: int = 3
    spec_metrics: SpecMetrics | None = None
    running_reqs: Set[Req] = field(default_factory=set)
    # Fair alternation when greedy (verify) and sampled (decode) sets both nonempty.
    _schedule_greedy_next: bool = field(default=True, init=False)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = sorted(self.running_reqs, key=lambda req: req.uid)

        # Spec off: byte-identical to the original decode path.
        if self.spec_algorithm == "none":
            return Batch(reqs=reqs, phase="decode")

        # Verify takes argmax over all K+1 rows, so sampled reqs must never share a
        # verify batch. Fair-alternate when both sets are non-empty to avoid starvation.
        greedy = [req for req in reqs if req.sampling_params.is_greedy]
        sampled = [req for req in reqs if not req.sampling_params.is_greedy]
        if greedy and sampled:
            selected = greedy if self._schedule_greedy_next else sampled
            self._schedule_greedy_next = not self._schedule_greedy_next
        else:
            selected = greedy or sampled

        if selected and not selected[0].sampling_params.is_greedy:
            return Batch(reqs=selected, phase="decode")

        # Draft on CPU, then emit phase="verify" (K+1 mini-extend). If nobody drafts,
        # fall back to phase="decode" so we keep the CUDA-graph path.
        any_draft = False
        for req in selected:
            req.draft_tokens.clear()
            # Leave one slot for the guaranteed bonus/correction token.
            max_draft = min(self.spec_num_draft, max(0, req.remain_len - 1))
            req.draft_tokens.extend(
                propose_ngram(
                    req.input_ids,
                    min_ngram=self.spec_ngram_min,
                    max_ngram=self.spec_ngram_max,
                    max_tokens=max_draft,
                )
            )
            if self.spec_metrics is not None:
                self.spec_metrics.record_proposal(bool(req.draft_tokens))
            any_draft |= bool(req.draft_tokens)
        return Batch(reqs=selected, phase="verify" if any_draft else "decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
