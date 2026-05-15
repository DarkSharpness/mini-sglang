from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Set, Dict

from minisgl.core import Batch, Req

SKIP_PROB = 0.3

@dataclass
class DecodeManager:
    page_size: int
    # blocks of model
    num_blocks: int
    # maximum bach sizes supported by our cuda graphs
    max_graph_bs: int
    # virtual pipeline queues
    virtual_queues: Dict[int, Set[Req]] = field(default_factory=dict)

    def __post_init__(self):
        self.virtual_queues = {i: set() for i in range(self.num_blocks)}
    
    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        for req in reqs:
            if req.can_decode:
                # place the request in the queue for its current block
                self.virtual_queues[req.current_block].add(req)

    def remove_req(self, req: Req) -> None:
        for q in self.virtual_queues.values():
            q.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for q in self.virtual_queues.values():
            for req in list(q):
                if req.uid == uid:
                    q.remove(req)
                    return req
        return None

    @property
    def inflight_tokens(self) -> int:
        count = 0
        for q in self.virtual_queues.values():
            tokens_reserved = (self.page_size - 1) * len(q)
            count += sum(req.remain_len for req in q) + tokens_reserved
        return count

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        
        # deepest ready first scheduling
        for block_idx in reversed(range(self.num_blocks)):
            reqs_in_queue = self.virtual_queues[block_idx]
            if not reqs_in_queue:
                continue
                
            # get requests waiting for this block, up to the max batch size
            batch_reqs = list(reqs_in_queue)[:self.max_graph_bs]

            # TODO: this is a simple router that skips with some probability
            if block_idx == 0:
                # we never skip the first block
                is_project = False
            else:
                is_project = random.random() < SKIP_PROB
                
            # remove them from the queue, as they are now in-flight on the GPU
            for req in batch_reqs:
                self.virtual_queues[block_idx].remove(req)
            
            batch = Batch(reqs=batch_reqs, phase="decode")
            batch.block_idx = block_idx
            batch.is_project = is_project
            return batch
            
        return None

    @property
    def runnable(self) -> bool:
        return any(len(q) > 0 for q in self.virtual_queues.values())
