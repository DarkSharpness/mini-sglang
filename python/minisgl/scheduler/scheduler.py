from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
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
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    new_2d_indices: torch.Tensor
    out_2d_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        self.tp_info = config.tp_info
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager(self.cache_manager, self.table_manager)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        self.device_id_pool = torch.empty_like(self.engine.page_table, dtype=torch.int32)

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply = BatchTokenizerMsg(data=[])

        for i, req in enumerate(batch.reqs):
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

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.host_ids[: req.cached_len],
                self.engine.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            self.prefill_manager.add_raw_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _schedule_next_batch(self) -> ForwardInput | None:
        from minisgl.kernel import make_2d_indices

        # TODO: support other policies: e.g. DECODE first
        prefill_budget = self.config.max_extend_tokens
        result = (
            self.prefill_manager.schedule_next_batch(prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if result is None:
            return None
        batch = Batch(reqs=result[1])
        self.engine.prepare_batch(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            new_2d_indices=result[0],
            out_2d_indices=make_2d_indices(
                self.table_manager.token_pool,
                ranges=[(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs],
                load_table=False,
            ),
        )

    def _load_token_ids(self, input: ForwardInput) -> None:
        # NOTE: this function must be called in the engine's forward stream
        batch, new_2d_indices = input.batch, input.new_2d_indices
        new_tokens = len(new_2d_indices)
        padded_new_tokens = new_tokens + (batch.padded_size - batch.size)
        batch.input_ids = torch.empty(padded_new_tokens, dtype=torch.int32, device=self.device)
        batch.input_ids[:new_tokens] = self.table_manager.token_pool.view(-1)[new_2d_indices]
        batch.input_ids[new_tokens:].zero_()

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        # NOTE: this function must be called in the engine's forward stream
        self.table_manager.token_pool.view(-1)[input.out_2d_indices] = output.next_tokens_gpu

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.critical_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        assert torch.cuda.current_stream() == self.stream
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # schedule this batch
        forward_input = self._schedule_next_batch()

        # run the batch in the engine's forward stream
        # we only process the metadata in the scheduler stream
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = None
            if forward_input is not None:
                self._load_token_ids(forward_input)
                batch, sample_args = forward_input.batch, forward_input.sample_args
                forward_output = self.engine.forward_batch(batch, sample_args)
                self._write_token_ids(forward_input, forward_output)
                self.decode_manager.add_reqs(forward_input.batch.reqs)
                ongoing_data = (forward_input, forward_output)

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def run_forever(self) -> NoReturn:
        ongoing_data = None
        while True:
            ongoing_data = self.overlap_loop(ongoing_data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()
