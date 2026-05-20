from __future__ import annotations

import time
from concurrent import futures
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Set, Tuple, TypeAlias

import torch
from minisgl.constrained import INVALID_GRAMMAR_OBJ, create_grammar_backend
from minisgl.env import ENV

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.constrained import BaseGrammarObject, GrammarKey
    from transformers import PreTrainedTokenizerBase

GRAMMAR_JSON = "json"
GRAMMAR_READY = "ready"
GRAMMAR_FAILED = "failed"
GRAMMAR_QUEUED = "queued"
GrammarSubmitStatus: TypeAlias = Literal["ready", "failed", "queued"]


@dataclass
class GrammarPollResult:
    ready_reqs: List[PendingReq]
    failed_uids: List[int]


class GrammarManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        eos_token_id: int,
        tp_cpu_group: torch.distributed.ProcessGroup,
    ) -> None:
        self.grammar_queue: List[PendingReq] = []
        self.grammar_backend = create_grammar_backend(
            tokenizer,
            vocab_size,
            eos_token_id,
        )
        self.tp_cpu_group = tp_cpu_group
        self.tp_size = torch.distributed.get_world_size(group=tp_cpu_group)
        self.poll_interval = ENV.GRAMMAR_POLL_INTERVAL.value
        self.max_poll_iterations = ENV.GRAMMAR_MAX_POLL_ITERATIONS.value

    @property
    def runnable(self) -> bool:
        return len(self.grammar_queue) > 0

    def _get_grammar_key(self, req: PendingReq) -> GrammarKey:
        if (json_schema := req.sampling_params.json_schema) is not None:
            return (GRAMMAR_JSON, json_schema)
        raise NotImplementedError("Structured output format is not implemented")

    def submit(self, req: PendingReq) -> GrammarSubmitStatus:
        assert req.constraint is not None
        key = self._get_grammar_key(req)
        req.constraint.grammar_key = key

        backend = self.grammar_backend
        if backend is None:
            req.constraint.grammar = None
            return GRAMMAR_FAILED

        value, cache_hit = backend.get_cached_or_future_value(key)
        req.constraint.grammar = value
        if cache_hit:
            if value is INVALID_GRAMMAR_OBJ:
                req.constraint.grammar = None
                return GRAMMAR_FAILED
            return GRAMMAR_READY

        self.grammar_queue.append(req)
        return GRAMMAR_QUEUED

    def poll_ready(self) -> GrammarPollResult:
        ready_uids: Set[int] = set()
        failed_uids: Set[int] = set()
        ready_values: dict[int, BaseGrammarObject] = {}

        deadline = time.perf_counter() + self.poll_interval
        while True:
            timeout = time.perf_counter() >= deadline
            for req in self.grammar_queue:
                uid = req.uid
                if uid in ready_uids or uid in failed_uids:
                    continue

                assert req.constraint is not None
                grammar = req.constraint.grammar
                assert isinstance(grammar, futures.Future)
                if grammar.done():
                    value = grammar.result()
                    if value is INVALID_GRAMMAR_OBJ:
                        failed_uids.add(uid)
                    else:
                        ready_uids.add(uid)
                        ready_values[uid] = value
                elif timeout:
                    req.constraint.grammar_wait_ct += 1
                    if req.constraint.grammar_wait_ct >= self.max_poll_iterations:
                        failed_uids.add(uid)

            if timeout:
                break
            time.sleep(self.poll_interval / 10)

        if self.tp_size > 1:
            gathered: List[Tuple[Set[int], Set[int]] | None] = [None] * self.tp_size
            torch.distributed.all_gather_object(
                gathered,
                (ready_uids, failed_uids),
                group=self.tp_cpu_group,
            )
            ready_uids = set.intersection(*(x[0] for x in gathered if x is not None))
            failed_uids = set.union(*(x[1] for x in gathered if x is not None))

        ready_uids -= failed_uids
        backend = self.grammar_backend
        ready_reqs: List[PendingReq] = []
        failed_list: List[int] = []
        next_queue: List[PendingReq] = []

        for req in self.grammar_queue:
            uid = req.uid
            assert req.constraint is not None
            key = req.constraint.grammar_key
            grammar = req.constraint.grammar

            if uid in failed_uids:
                if isinstance(grammar, futures.Future):
                    grammar.cancel()
                if backend is not None and key is not None:
                    backend.set_cache(key, INVALID_GRAMMAR_OBJ)
                req.constraint.grammar = None
                failed_list.append(uid)
                continue

            if uid in ready_uids:
                value = ready_values[uid]
                req.constraint.grammar = value
                if backend is not None and key is not None:
                    backend.set_cache(key, value.copy())
                ready_reqs.append(req)
                continue

            next_queue.append(req)

        self.grammar_queue = next_queue
        return GrammarPollResult(ready_reqs=ready_reqs, failed_uids=failed_list)

    def abort_req(self, uid: int) -> bool:
        for i, req in enumerate(self.grammar_queue):
            if req.uid != uid:
                continue
            grammar = req.constraint.grammar
            if isinstance(grammar, futures.Future):
                grammar.cancel()
            self.grammar_queue.pop(i)
            return True
        return False

    def shutdown(self) -> None:
        for req in self.grammar_queue:
            grammar = req.constraint.grammar
            if isinstance(grammar, futures.Future):
                grammar.cancel()
        self.grammar_queue.clear()
        if self.grammar_backend is not None:
            self.grammar_backend.shutdown()
