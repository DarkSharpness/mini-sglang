from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import call_if_main, init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    from minisgl.scheduler import Scheduler

    scheduler = Scheduler(config=args)
    scheduler.sync_all_ranks()

    if args.tp_info.is_primary():
        ack_queue.put("Scheduler is ready")

    scheduler.run_forever()


@call_if_main(__name__, discard=False)
def launch_server() -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args = parse_args(sys.argv[1:])
    logger = init_logger(__name__)

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
            ).start()

        if server_args.share_tokenizer:
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_unique_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": 0,
                    "ack_queue": ack_queue,
                },
                daemon=False,
            ).start()
            num_tokenizers = 1
        else:
            assert False, "Not implemented"

        # 1 scheduler + num_tokenizers
        for _ in range(num_tokenizers + 1):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess)
