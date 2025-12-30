from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer
import pytest

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from minisgl.core import SamplingParams

logger = init_logger(__name__)


@torch.inference_mode()
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    scheduler = Scheduler(config)
    queue.put(None)
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


@pytest.mark.e2e
def test_generation():
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[2, 4, 8],
    )

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler, args=(config, q))
    p.start()
    q.get()

    send_backend = ZmqPushQueue(
        config.zmq_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )

    recv_backend = ZmqPullQueue(
        config.zmq_detokenizer_addr,
        create=False,
        decoder=BaseTokenizerMsg.decoder,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    send_backend.put(
        UserMsg(
            uid=0,
            input_ids=ids,
            sampling_params=SamplingParams(max_tokens=100),
        )
    )

    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:
            break

    logger.info(tokenizer.decode(ids.tolist()))
    send_backend.put(ExitMsg())
    p.join()
