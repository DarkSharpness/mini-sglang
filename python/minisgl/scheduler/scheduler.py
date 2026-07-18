from __future__ import annotations

import json
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
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .spec import SpecMetrics, resolve_verify
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


class VerifyOutcome(NamedTuple):
    req: Req
    committed: List[int]
    finished: bool
    verify_end: int  # prior forward_device_len; rollback frees [cached_len, verify_end)


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        _validate_spec_config(config)
        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.spec_metrics = SpecMetrics()
        self.decode_manager = DecodeManager(
            page_size=config.page_size,
            # Spec knobs live on the decode scheduler (engine only keys off Batch.phase).
            spec_algorithm=config.spec_algorithm,
            spec_num_draft=config.spec_num_draft,
            spec_ngram_min=config.spec_ngram_min,
            spec_ngram_max=config.spec_ngram_max,
            spec_metrics=self.spec_metrics,
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self._last_logged_spec_proposals = 0
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        self._log_spec_metrics()

    def _log_spec_metrics(self) -> None:
        if self.spec_metrics.proposal_requests != self._last_logged_spec_proposals:
            logger.info_rank0("SPEC_METRICS %s", json.dumps(self.spec_metrics.as_dict()))
            self._last_logged_spec_proposals = self.spec_metrics.proposal_requests

    def _log_spec_metrics_if_drained(self) -> None:
        if not self.prefill_manager.runnable and not self.decode_manager.runnable:
            self._log_spec_metrics()

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
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        # Verify has its own accept/commit/rollback path (variable 1..K+1 tokens).
        if batch.is_verify:
            self._process_verify_batch(batch, next_tokens_cpu)
            return

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
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                # Verify is not prefill ⇒ drafts never enter the radix cache here.
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Flush before the final response: client completion then guarantees that
        # cumulative spec metrics for this drained workload are already logged.
        self._log_spec_metrics_if_drained()
        self.send_result(reply)

    def _process_verify_batch(self, batch: Batch, target_tokens: torch.Tensor) -> None:
        """Accept drafts, commit 1..K+1 tokens, free rejected KV, stream detok msgs."""
        outcomes: List[VerifyOutcome] = []
        # Next decode/verify step needs the frontier token at the new cached_len.
        anchors: List[Tuple[int, int, int]] = []  # (table_idx, cached_len, token)
        # target_tokens is flat over the batch: K+1 argmax rows per req (ragged).
        offset = 0
        with self.cache_manager.lazy_free_region():
            for req in batch.reqs:
                num_rows = req.extend_len  # 1 + len(draft_tokens); via forward_device_len
                targets = target_tokens[offset : offset + num_rows].tolist()
                offset += num_rows
                # accept → truncate EOS/max_tokens → advance state (rollback batched below)
                outcome = self._commit_verify_req(req, targets, num_draft=num_rows - 1)
                outcomes.append(outcome)
                if not outcome.finished:
                    anchors.append((req.table_idx, req.cached_len, outcome.committed[-1]))

            assert offset == len(target_tokens)
            # One free for all rejected draft slots
            self.cache_manager.rollback_paged_batch(
                [(o.req.table_idx, o.req.cached_len, o.verify_end) for o in outcomes]
            )
            self._write_frontier_anchors(anchors)
            reply = self._finish_verify_outcomes(batch, outcomes)

        # Same drain contract as the normal path; no benchmark-side sleep needed.
        self._log_spec_metrics_if_drained()
        self.send_result(reply)

    def _commit_verify_req(
        self, req: Req, targets: List[int], *, num_draft: int
    ) -> VerifyOutcome:
        # Verifier accept count is unclamped (engine-style); only committed is truncated.
        resolved = resolve_verify(
            req.draft_tokens,
            targets,
            remain_len=req.remain_len,
            eos_token_id=self.eos_token_id,
            ignore_eos=req.sampling_params.ignore_eos,
        )
        # Capture before clearing drafts — forward_device_len includes the draft span.
        verify_end = req.forward_device_len
        num_final = len(resolved.committed)
        req.append_host(torch.tensor(resolved.committed, dtype=torch.int32))
        # Commit: cached_len covers accepted tokens; device_len parks the new frontier slot.
        # State dry run, K=3 and one accepted draft:
        # before: KV=[0,c), frontier x0@c, drafts d1..d3@c+1..c+3
        # verify: [x0,d1,d2,d3] -> [g0,g1,g2,g3], d1==g0, d2!=g1
        # commit: [g0,g1], KV=[0,c+2), frontier g1@c+2
        # rollback (batched): free stale draft KV at [c+2,c+4)
        req.cached_len += num_final
        req.device_len = req.cached_len + 1
        req.draft_tokens.clear()

        # accepted = verifier drafts matched (bonus/correction token is not a draft).
        self.spec_metrics.record(
            num_draft=num_draft,
            num_accepted=resolved.num_accepted,
        )
        return VerifyOutcome(req, resolved.committed, resolved.finished, verify_end)

    def _write_frontier_anchors(self, anchors: List[Tuple[int, int, int]]) -> None:
        # Decode/verify next step reads token_pool[table_idx, cached_len] as the frontier.
        if not anchors:
            return
        rows = torch.tensor([a[0] for a in anchors], dtype=torch.int64, pin_memory=True)
        cols = torch.tensor([a[1] for a in anchors], dtype=torch.int64, pin_memory=True)
        tokens = torch.tensor([a[2] for a in anchors], dtype=torch.int32, pin_memory=True)
        self.token_pool[
            rows.to(self.device, non_blocking=True),
            cols.to(self.device, non_blocking=True),
        ] = tokens.to(self.device, non_blocking=True)

    def _finish_verify_outcomes(
        self, batch: Batch, outcomes: List[VerifyOutcome]
    ) -> List[DetokenizeMsg]:
        # Drop finished reqs from the decode set; one DetokenizeMsg per committed token.
        self.decode_manager.filter_reqs(batch.reqs)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        for req, committed, finished, _verify_end in outcomes:
            for i, token in enumerate(committed):
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=token,
                        finished=finished and i == len(committed) - 1,
                    )
                )
            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                # Finished ⇒ radix-insert committed tokens only (drafts already cleared).
                self._free_req_resources(req)
                new_finished_reqs.add(req)
        self.finished_reqs = new_finished_reqs
        return reply

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
        self.engine.graph_runner.pad_batch(batch)
        # Write drafts into the token pool before KV alloc / position / FI metadata
        # so the verify forward sees [last_committed, draft_1..draft_K].
        if batch.is_verify:
            self._stage_drafts(batch)
        # allocate_paged / positions / FI metadata all read forward_device_len
        # (includes drafts for verify; == device_len otherwise).
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _stage_drafts(self, batch: Batch) -> None:
        """Lay out draft tokens after the committed frontier for the verify forward."""
        assert all(req.sampling_params.is_greedy for req in batch.reqs)
        # One advanced-index H2D write for the whole batch (vs N tiny copies).
        rows: List[int] = []
        cols: List[int] = []
        tokens: List[int] = []
        for req in batch.reqs:
            if not req.draft_tokens:
                continue
            # device_len is the frontier slot; drafts occupy [device_len, forward_device_len).
            start = req.device_len
            n = len(req.draft_tokens)
            rows.extend([req.table_idx] * n)
            cols.extend(range(start, start + n))
            tokens.extend(req.draft_tokens)
        if not tokens:
            return
        rows_t = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
        cols_t = torch.tensor(cols, dtype=torch.int64, pin_memory=True)
        toks_t = torch.tensor(tokens, dtype=torch.int32, pin_memory=True)
        self.token_pool[
            rows_t.to(self.device, non_blocking=True),
            cols_t.to(self.device, non_blocking=True),
        ] = toks_t.to(self.device, non_blocking=True)

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
        # Verify writes the frontier later in _process_verify_batch (after accept).
        if not batch.is_verify:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # Uses forward_device_len so verify positions cover the draft span too.
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.forward_device_len,
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
    # Decode path writes the sampled token at device_len. Verify skips this write
    # (see _forward) and plants the frontier later via _write_frontier_anchors.
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)


def _validate_spec_config(config: SchedulerConfig) -> None:
    # Base-scope gates: FI page_size=1 + TP=1 + overlap off. Accept length is
    # data-dependent, so normal_loop is correct-by-construction; overlap is a follow-up.
    if config.spec_algorithm == "none":
        return
    if config.spec_algorithm != "ngram":
        raise ValueError(f"Unsupported speculative algorithm: {config.spec_algorithm}")
    if config.tp_info.size != 1:
        raise ValueError("N-gram speculation currently requires TP=1.")
    if config.page_size != 1:
        raise ValueError("N-gram speculation currently requires page_size=1.")
    if config.attention_backend != "fi":
        raise ValueError("N-gram speculation currently requires --attention-backend fi.")
    if not ENV.DISABLE_OVERLAP_SCHEDULING:
        raise ValueError(
            "N-gram speculation currently requires MINISGL_DISABLE_OVERLAP_SCHEDULING=1."
        )
    if config.spec_num_draft < 1:
        raise ValueError("spec_num_draft must be at least 1.")
    if not 1 <= config.spec_ngram_min <= config.spec_ngram_max:
        raise ValueError("Expected 1 <= spec_ngram_min <= spec_ngram_max.")
