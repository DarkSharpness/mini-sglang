from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Set

from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.utils import alloc_delta

if TYPE_CHECKING:
    from .cache import CacheManager
    from .prefill import PrefillManager
    from .table import TableManager


@dataclass
class _EstimatePolicy:
    page_size: int
    init_new_token_ratio: float = field(init=False)
    min_new_token_ratio: float = field(init=False)
    new_token_ratio_decay: float = field(init=False)
    new_token_ratio: float = field(init=False)
    clip_max_new_tokens: int = field(init=False)
    retract_decode_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.init_new_token_ratio = min(
            ENV.INIT_NEW_TOKEN_RATIO.value * ENV.SCHEDULE_CONSERVATIVENESS.value, 1.0
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * ENV.MIN_NEW_TOKEN_RATIO_FACTOR.value, 1.0
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / ENV.NEW_TOKEN_RATIO_DECAY_STEPS.value
        self.new_token_ratio = self.init_new_token_ratio
        self.clip_max_new_tokens = ENV.CLIP_MAX_NEW_TOKENS.value
        self.retract_decode_steps = ENV.RETRACT_DECODE_STEPS.value

    def reset(self) -> None:
        self.new_token_ratio = self.init_new_token_ratio

    def estimated_inflight_tokens(self, reqs: Iterable[Req]) -> int:
        reserved_size = 0
        for req in reqs:
            if req.sampling_params.ignore_eos:
                tail_est = req.remain_len
            else:
                tail_est = math.ceil(
                    min(req.remain_len, self.clip_max_new_tokens) * self.new_token_ratio
                )
            reserved_size += alloc_delta(
                req.cached_len, req.extend_len + tail_est, self.page_size
            )
        return reserved_size

    def on_decode_success(self) -> None:
        self.new_token_ratio = max(
            self.min_new_token_ratio,
            self.new_token_ratio - self.new_token_ratio_decay,
        )

    def on_retract(self, reqs: Iterable[Req]) -> None:
        reqs = list(reqs)
        decoded_tokens = sum(len(req.input_ids) - req.prompt_len for req in reqs)
        total_tokens = sum(req.max_device_len - req.prompt_len for req in reqs)
        self.new_token_ratio = min(
            1.0,
            (decoded_tokens + self.retract_decode_steps * len(reqs)) / total_tokens,
        )


@dataclass
class DecodeManager:
    EstimatePolicy = _EstimatePolicy

    page_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    running_reqs: Set[Req] = field(default_factory=set)
    estimate_policy: "DecodeManager.EstimatePolicy" = field(init=False)

    def __post_init__(self) -> None:
        self.estimate_policy = self.EstimatePolicy(self.page_size)

    def reset_new_token_ratio(self) -> None:
        self.estimate_policy.reset()

    @property
    def clip_max_new_tokens(self) -> int:
        return self.estimate_policy.clip_max_new_tokens

    @property
    def estimated_inflight_tokens(self) -> int:
        return self.estimate_policy.estimated_inflight_tokens(self.running_reqs)

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

    def _check_decode_mem(self, available_size: int, steps: int = 1) -> bool:
        need = sum(
            alloc_delta(
                req.cached_len,
                min(req.remain_len + req.extend_len, steps),
                self.page_size,
            )
            for req in self.running_reqs
        )
        return need <= available_size

    def schedule_next_batch(
        self,
        prefill_manager: PrefillManager,
    ) -> Batch | None:
        if not self.runnable:
            return None
        if self._check_decode_mem(self.cache_manager.available_size):
            self.estimate_policy.on_decode_success()
            return Batch(reqs=list(self.running_reqs), phase="decode")

        retracted_reqs = []
        while not self._check_decode_mem(
            self.cache_manager.available_size,
            steps=self.estimate_policy.retract_decode_steps,
        ):
            if len(self.running_reqs) == 1:
                req = next(iter(self.running_reqs))
                raise RuntimeError(
                    f"Decode OOM ! retract_decode_steps={self.estimate_policy.retract_decode_steps}, "
                    f"cached_len={req.cached_len}"
                )
            req = min(
                self.running_reqs,
                key=lambda req: (len(req.input_ids) - req.prompt_len, -req.prompt_len, req.uid),
            )
            self.running_reqs.remove(req)
            req.is_retracted = True
            self.table_manager.free(req.table_idx)
            self.cache_manager.cache_req(req, finished=True)
            retracted_reqs.append(req)
        prefill_manager.requeue_reqs(retracted_reqs)

        batch = Batch(reqs=list(self.running_reqs), phase="decode")
        self.estimate_policy.on_retract(batch.reqs)
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
