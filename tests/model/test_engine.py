from __future__ import annotations

import torch

from minisgl.config.context import Batch, Req
from minisgl.config.engine import EngineConfig
from minisgl.distributed import DistributedInfo
from minisgl.engine.engine import Engine
from minisgl.utils import call_if_main


@call_if_main()
def main():
    config = EngineConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
    )
    engine = Engine(config)
    batch = Batch(
        reqs=[
            Req(
                input_ids=[0],
                page_table_idx=0,
                cached_len=0,
                output_len=10,
                device=engine.device,
                rid=0,
            )
        ],
    )
    engine.prepare_batch(batch)
    engine.forward_batch(batch)
