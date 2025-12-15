from __future__ import annotations

from typing import Iterable, Set

from minisgl.core import Batch, Req


class DecodeManager:
    def __init__(self):
        self.running_reqs: Set[Req] = set()

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def remove_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.difference_update(reqs)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        from .prefill import ChunkedReq

        assert not any(isinstance(r, ChunkedReq) for r in self.running_reqs)
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)
