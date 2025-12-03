from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Set

import torch
from minisgl.config.context import Batch, Req
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import PageTableManager

if TYPE_CHECKING:
    from minisgl.engine.engine import EngineResult

logger = init_logger(__name__)


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        self.tp_info = config.tp_info
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.stream = torch.cuda.Stream()
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        self.this_batch = None
        self.table_manager = PageTableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.device, self.engine.num_pages, config.cache_type
        )
        self.decode_manager = DecodeManager(self.cache_manager, self.table_manager)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id

    def _process_batch_result(
        self, last_result: EngineResult, last_batch: Batch
    ) -> BatchTokenizerMsg:
        next_tokens_cpu = last_result.next_tokens_cpu
        reply = BatchTokenizerMsg(data=[])

        for i, req in enumerate(last_batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())
            finished = req.remain_len <= 0
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        ongoing_reqs = self.this_batch.reqs if self.this_batch else []

        # free resources for finished but not ongoing reqs
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.page_table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.host_ids[: req.cached_len],
                self.engine.page_table[req.page_table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        return reply

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # TODO: graceful shutdown
            self.engine.shutdown()
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            self.prefill_manager.add_raw_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _schedule_next_batch(self) -> Batch | None:
        # TODO: support other policies: e.g. DECODE first
        prefill_budget = self.config.max_extend_tokens
        if result := self.prefill_manager.schedule_next_batch(prefill_budget):
            return Batch(reqs=result)
        if result := self.decode_manager.schedule_next_batch():
            return Batch(reqs=result)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.critical_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def overlap_loop(self) -> None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of this batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        assert torch.cuda.current_stream() == self.stream
        last_batch = self.this_batch
        last_result = self.engine.last_batch_result
        self.this_batch = None

        blocking = not (
            last_batch  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # schedule this batch
        this_batch = self.this_batch = self._schedule_next_batch()
        if this_batch is not None:
            self.engine.prepare_batch(this_batch)

        # run the batch in the engine's forward stream
        # we only process the metadata in the scheduler stream
        last_result.onboard_event.synchronize()
        last_result.onboard_event.record(self.stream)

        with self.engine_stream_ctx:
            last_result.onboard_event.wait(self.engine.stream)
            if this_batch is not None:
                logger.debug_rank0(f"Running a {this_batch._phase.capitalize()} batch")
                self.engine.forward_batch(this_batch)
                self.decode_manager.add_reqs(
                    req for req in this_batch.reqs if not isinstance(req, ChunkedReq)
                )

        # after schedule
        if last_batch is None:
            return

        last_result.offload_event.synchronize()
        self.send_result(self._process_batch_result(last_result, last_batch))

    def run_forever(self) -> NoReturn:
        while True:
            self.overlap_loop()
