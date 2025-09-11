from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer

logger = init_logger(__name__, "tokenizer")


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    from .detokenizer import DetokenizeManager
    from .tokenizer import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenizer server {tokenizer_id} is ready")

    while True:
        pending_msg = _unwrap_msg(recv_listener.get())
        while len(pending_msg) < local_bs and not recv_listener.empty():
            pending_msg.extend(_unwrap_msg(recv_listener.get()))

        logger.debug(f"Received {len(pending_msg)} messages")

        detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
        tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
        assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)
        if len(detokenize_msg) > 0:
            replies = detokenize_manager.detokenize(detokenize_msg)
            batch_output = BatchFrontendMsg(
                data=[
                    UserReply(
                        uid=msg.uid,
                        incremental_output=reply,
                        finished=msg.finished,
                    )
                    for msg, reply in zip(detokenize_msg, replies, strict=True)
                ]
            )
            if len(batch_output.data) == 1:
                batch_output = batch_output.data[0]
            send_frontend.put(batch_output)

        if len(tokenize_msg) > 0:
            tensors = tokenize_manager.tokenize(tokenize_msg)
            batch_output = BatchBackendMsg(
                data=[
                    UserMsg(
                        uid=msg.uid,
                        input_ids=t,
                        output_len=msg.output_len,
                    )
                    for msg, t in zip(tokenize_msg, tensors, strict=True)
                ]
            )
            if len(batch_output.data) == 1:
                batch_output = batch_output.data[0]
            send_backend.put(batch_output)
