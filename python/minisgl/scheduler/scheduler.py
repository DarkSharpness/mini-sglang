from __future__ import annotations

from typing import TYPE_CHECKING, List, NoReturn, Set

import torch
from minisgl.config.context import Batch, Req
from minisgl.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from minisgl.utils.mp import ZmqPubQueue, ZmqSubQueue

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .prefill import PrefillManager
from .table import PageTableManager

if TYPE_CHECKING:
    from minisgl.engine.engine import EngineResult

logger = init_logger(__name__)


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.config = config
        self.engine = Engine(config)
        self.tp_info = config.tp_info

        # use another stream to overlap metadata processing with computation
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)

        self.tp_cpu_group = self.engine.tp_cpu_group

        self.recv_msg = self._recv_msg_single_rank
        self.reply_tokenizer = self._reply_tokenizer_rank0

        if self.tp_info.is_primary():
            # queues for receiving/sending data to/from tokenizer
            self.recv_tokenizer = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self.send_tokenizer = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        if self.tp_info.size > 1:
            if self.tp_info.is_primary():
                self.recv_msg = self._recv_msg_multi_rank0
                self.send_ranks = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
                self.sync_all_ranks()
            else:
                self.recv_msg = self._recv_msg_multi_rank1
                self.reply_tokenizer = self._reply_tokenizer_rank1
                self.sync_all_ranks()
                self.recv_ranks = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # just to make sure all queues are created
        self.sync_all_ranks()

        self.last_batch = None
        self.table_manager = PageTableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.engine.device, self.engine.num_pages)
        self.decode_manager = DecodeManager(self.cache_manager, self.table_manager)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            logger.debug_rank0("Waiting for messages from tokenizer...")
            pending_msgs.append(self.recv_tokenizer.get())
        while not self.recv_tokenizer.empty():
            pending_msgs.append(self.recv_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        pending_raw_msgs: List[bytes] = []
        if blocking:
            pending_raw_msgs.append(self.recv_tokenizer.get_raw())
        while not self.recv_tokenizer.empty():
            pending_raw_msgs.append(self.recv_tokenizer.get_raw())
        for raw in pending_raw_msgs:
            self.send_ranks.put_raw(raw)
            msg = self.recv_tokenizer.decode(raw)
            pending_msgs.append(msg)
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            pending_msgs.append(self.recv_ranks.get())
        while not self.recv_ranks.empty():
            pending_msgs.append(self.recv_ranks.get())
        return pending_msgs

    def _filter_finished_reqs(
        self, last_result: EngineResult, last_batch: Batch
    ) -> BatchTokenizerMsg:
        next_tokens_cpu = last_result.next_tokens_cpu
        reply = BatchTokenizerMsg(data=[])
        for i, req in enumerate(last_batch.reqs):
            if req in self.finished_reqs:
                continue

            next_token = int(next_tokens_cpu[i].item())
            # TODO: test whether finished by using eos
            finished = req.remain_len <= 0
            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)

        ongoing_reqs = last_batch.reqs if last_batch else []

        # free resources for finished but not ongoing reqs
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.page_table_idx)
            self.cache_manager.free(
                req.cache_handle,
                req.device_ids,
                self.engine.page_table[req.page_table_idx, : req.device_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        return reply

    def _reply_tokenizer_rank0(self, last_result: EngineResult, last_batch: Batch) -> None:
        reply = self._filter_finished_reqs(last_result, last_batch)
        num_reply = len(reply.data)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self.send_tokenizer.put(reply.data[0])
        elif num_reply > 1:
            self.send_tokenizer.put(reply)

    def _reply_tokenizer_rank1(self, last_result: EngineResult, last_batch: Batch) -> None:
        reply = self._filter_finished_reqs(last_result, last_batch)
        _ = reply  # do nothing for non-primary ranks

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # TODO: graceful shutdown
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0(f"Received user msg: {msg}")
            self.prefill_manager.add_raw_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def sync_all_ranks(self) -> None:
        if self.tp_info.size > 1:
            self.tp_cpu_group.barrier().wait()

    def _schedule_next_batch(self) -> Batch | None:
        # TODO: support other policies: e.g. DECODE first
        prefill_budget = self.config.max_extend_tokens
        if result := self.prefill_manager.schedule_next_batch(prefill_budget):
            return Batch(reqs=result)
        if result := self.decode_manager.schedule_next_batch():
            return Batch(reqs=result)

    @torch.inference_mode()
    def main_loop(self) -> None:
        assert torch.cuda.current_stream() == self.stream
        logger.debug_rank0("Scheduler main loop iteration")
        last_batch = self.last_batch
        last_result = self.engine.last_batch_result
        self.last_batch = None

        blocking = not (
            self.last_batch  # don't block if we have a batch to run
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        if blocking:
            logger.debug_rank0("No runnable batch, blocking for new messages")

        for msg in self.recv_msg(blocking=blocking):
            self._process_one_msg(msg)

        # schedule this batch
        self.last_batch = this_batch = self._schedule_next_batch()

        # run the batch in the engine's forward stream
        # we only process the metadata in the scheduler stream
        last_result.onboard_event.synchronize()
        last_result.onboard_event.record(self.stream)
        with torch.cuda.stream(self.engine.stream):
            last_result.onboard_event.wait(self.engine.stream)
            if this_batch is not None:
                logger.debug_rank0(f"Running a {this_batch._phase.capitalize()} batch")
                self.engine.prepare_batch(this_batch)
                self.engine.forward_batch(this_batch)
                self.decode_manager.add_reqs(this_batch.reqs)

        # after schedule
        if last_batch is None:
            return

        logger.debug_rank0("Processing last batch results")
        last_result.offload_event.synchronize()
        self.reply_tokenizer(last_result, last_batch)

    def run_forever(self) -> NoReturn:
        while True:
            self.main_loop()
