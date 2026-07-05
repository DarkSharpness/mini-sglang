from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import div_ceil, init_logger, load_tokenizer
from minisgl.utils.device_runtime import (
    create_stream,
    current_stream,
    set_stream,
    stream_context,
    synchronize_device,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        # Stream lifecycle routes through the shared device_runtime dispatch so
        # cuda / npu / cpu hosts all follow the same call graph. Engine already
        # committed to this layer for its own stream — Scheduler reuses the
        # engine's device_type so the two managers can never drift apart.
        self.device_type = self.engine.device_type
        self.stream = create_stream(self.device_type)
        self.engine_stream_ctx = stream_context(self.device_type, self.engine.stream)
        set_stream(self.device_type, self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert current_stream(self.device_type) == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        synchronize_device(self.device_type)
        self.sync_all_ranks()
        # Gate 2.3d: proactively drain pending + running requests BEFORE
        # engine.shutdown() so the allocator/table_manager are restored to
        # baseline. Engine shutdown itself only tears down comm groups and
        # graph capture — it does not walk the scheduler's queues.
        self._drain_requests()
        self.engine.shutdown()

    def _drain_requests(self) -> None:
        """Release every request currently owned by this scheduler.

        Called from ``shutdown()`` between the device/rank sync and the
        engine teardown. Guarantees on return:

        * ``prefill_manager.pending_list`` is empty.
        * ``decode_manager.running_reqs`` is empty.
        * Every ``table_idx`` those requests held is back on
          ``table_manager._free_slots``.
        * Every non-cached KV page those requests owned is back on
          ``cache_manager.free_slots`` (via ``cache_req(..., finished=True)``).
        * ``cache_manager.check_integrity()`` passes.

        Idempotent: calling twice is a no-op — the second call finds both
        containers already empty and returns without touching the
        allocators.

        Deliberately out of scope (per Gate 2.3d):
          * No abort ack, no final DetokenizeMsg emission.
          * No handling of overlap-loop in-flight batches (those are
            drained by the earlier ``synchronize_device`` call).
        """
        # Dedup by object identity so a Req referenced from multiple
        # containers (e.g. the pending list's ChunkedReq AND the running
        # set, if bookkeeping ever drifts) is freed exactly once. Uid is
        # user-controlled and not guaranteed unique across the lifetime
        # of a scheduler; id() is.
        freed_ids: Set[int] = set()

        def _free_once(req: Req) -> None:
            if id(req) in freed_ids:
                return
            freed_ids.add(id(req))
            # Standard release path frees pages in [0, req.cached_len). For
            # mid-flight requests (chunked prefill that never advanced,
            # decode reqs stopped between forward and post-processing) the
            # region [cached_len, device_len) also holds physical KV pages
            # that allocate_paged wrote — those would otherwise leak. Mirror
            # allocate_paged's page math (round cached_len UP, device_len UP)
            # so we only touch page-aligned regions the allocator itself
            # produced. If they coincide (nothing was ever page-allocated for
            # this req beyond the prefix), the slice is empty and _free is a
            # no-op.
            first_page = div_ceil(req.cached_len, self.cache_manager.page_size)
            last_page = div_ceil(req.device_len, self.cache_manager.page_size)
            if last_page > first_page:
                first_pos = first_page * self.cache_manager.page_size
                last_pos = last_page * self.cache_manager.page_size
                tail_slots = self.cache_manager.page_table[
                    req.table_idx, first_pos:last_pos
                ]
                self.cache_manager._free(tail_slots)
            self._free_req_resources(req)

        # -- pending -----------------------------------------------------------
        # PendingReq is a lightweight envelope carrying user_ids /
        # sampling_params. If `.chunked_req is None`, no Req was ever built
        # for it — no table_idx, no KV pages — so dropping the entry is the
        # entire cleanup. If `.chunked_req is not None`, a real Req was
        # allocated (in PrefillAdder) with a table slot and possibly KV
        # pages; that one needs _free_req_resources exactly once.
        pending_list = self.prefill_manager.pending_list
        for pending in pending_list:
            chunked = pending.chunked_req
            if chunked is not None:
                _free_once(chunked)
        pending_list.clear()

        # -- running -----------------------------------------------------------
        # Snapshot to a list to iterate deterministically — a Set's iteration
        # order is nondeterministic across Python runs, and any future
        # dependency between free ordering and page reuse would silently
        # regress. Sorting by uid gives a stable, reviewable order.
        running_snapshot = sorted(self.decode_manager.running_reqs, key=lambda r: r.uid)
        for req in running_snapshot:
            _free_once(req)
        self.decode_manager.running_reqs.clear()

        # -- finished bookkeeping ---------------------------------------------
        # The overlap-scheduling second-free guard tracks recently-finished
        # reqs so a repeated free is a no-op. Once drained, that guard is
        # irrelevant — dump the set so a subsequent drain call cannot see a
        # phantom "already freed" mask.
        self.finished_reqs = set()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                # Gate 2.1c: explicit per-request stop tokens. Evaluated AFTER
                # req.append_host so the stop token itself is retained in the
                # output sequence (spec: "stop token 本身必须保留在输出序列中").
                # Runs independently of ignore_eos — ignore_eos silences only
                # the tokenizer's EOS, never the user-declared stop set. Empty
                # tuple keeps the pre-2.1c behavior a strict no-op.
                if next_token in req.sampling_params.stop_token_ids:
                    finished = True
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # Gate 2.3b: transactional page-table allocation. Everything from
        # ``allocate_paged`` onward mutates scratch state on ``batch``; any
        # raise before we return must undo *only* what this call added — new
        # physical pages and page_table slice overwrites — without touching
        # prefix pages, cache handles, or the running/pending sets.
        self.engine.graph_runner.pad_batch(batch)
        token = self.cache_manager.allocate_paged(batch.reqs)
        try:
            batch.positions = _make_positions(batch, self.device)
            input_mapping = _make_input_tuple(batch, self.device)
            write_mapping = _make_write_tuple(batch, self.device)
            batch.out_loc = self.engine.page_table[input_mapping]
            self.engine.attn_backend.prepare_metadata(batch)
            sample_args = self.engine.sampler.prepare(batch)
        except BaseException:
            # Roll back only the allocation performed above. Then drop every
            # scratch attribute the failed prepare wrote onto ``batch`` so a
            # later scheduling round cannot accidentally reuse a stale
            # positions / out_loc / metadata pointing at freed pages.
            self.cache_manager.rollback_allocation(token)
            for attr in ("positions", "out_loc", "padded_reqs", "attn_metadata"):
                if attr in batch.__dict__:
                    del batch.__dict__[attr]
            raise
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
