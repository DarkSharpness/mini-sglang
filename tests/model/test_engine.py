from __future__ import annotations

import torch
from transformers import AutoTokenizer

from minisgl.config.context import Batch, Req
from minisgl.config.engine import EngineConfig
from minisgl.distributed import DistributedInfo
from minisgl.engine.engine import Engine
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@call_if_main()
def main():
    config = EngineConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[2, 4, 8],
    )

    engine = Engine(config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"
    req = Req(
        input_ids=tokenizer.encode(prompt),
        page_table_idx=0,
        cached_len=0,
        output_len=10,
        device=engine.device,
        rid=0,
    )
    # allocate indices for page table
    engine.ctx.page_table[0][:1000] = torch.arange(1000, dtype=torch.int32, device=engine.device)

    for _ in range(100):
        batch = Batch(reqs=[req])
        engine.prepare_batch(batch)
        engine.forward_batch(batch)
        result = engine.last_batch_result
        result.offload_event.synchronize()
        next_token = int(result.next_tokens_cpu[0].item())
        if next_token == tokenizer.eos_token_id:
            break

    tokens = req.device_ids.cpu().tolist()
    logger.info_rank0(tokenizer.decode(tokens))
