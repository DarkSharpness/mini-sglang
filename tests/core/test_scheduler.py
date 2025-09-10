from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer

from minisgl.config.context import Batch, Req
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, ExitMsg, UserMsg
from minisgl.message.tokenizer import DetokenizeMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, call_if_main, init_logger


logger = init_logger(__name__)


def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    scheduler = Scheduler(config)
    queue.put(None)
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


@call_if_main(__name__)
def main():
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
        config.zmq_tokenizer_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )

    recv_backend = ZmqPullQueue(
        config.zmq_backend_tokenizer_addr,
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
            output_len=100,
        )
    )

    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:
            break

    print(tokenizer.decode(ids.tolist()))
    send_backend.put(ExitMsg())
