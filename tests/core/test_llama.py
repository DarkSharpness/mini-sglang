from __future__ import annotations
from typing import Any

import torch

from transformers import AutoConfig, AutoTokenizer
from minisgl.attention import create_attention_backend
from minisgl.config.context import Batch, Context, Req, set_global_ctx
from minisgl.models import ModelConfig
from minisgl.distributed import set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.models import load_hf_weight
from minisgl.models.llama import LlamaForCausalLM
from minisgl.utils import call_if_main, init_logger
from minisgl.utils.torch_utils import torch_dtype

logger = init_logger(__name__)


@call_if_main()
def main():
    set_tp_info(0, 1)
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    config: Any = AutoConfig.from_pretrained(model_path)
    model_config = ModelConfig.from_hf(config)
    logger.info(model_config, config)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    with torch_dtype(dtype), device:
        model = LlamaForCausalLM(model_config)

    model.load_state_dict(load_hf_weight(model_path, device))

    kv_cache = create_kvcache(
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_pages=65536,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )

    page_table = Context.create_page_table(256, 131072, device)
    attn_backend = create_attention_backend(model_config, kv_cache, "fa3", page_table)
    ctx = Context(
        page_table=page_table,
        page_size=1,
        kv_cache=kv_cache,
        attn_backend=attn_backend,
    )
    set_global_ctx(ctx)

    input_str = "Hello, what's your name?"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    batch = Batch(
        reqs=[
            Req(
                input_ids=tokenizer.encode(input_str),
                page_table_idx=1,
                cached_len=0,
                output_len=10,
                device=device,
                uid=0,
            )
        ],
    )

    attn_backend.prepare_metadata(batch, allow_graph=False)
    logits = model.forward_batch(batch)
    logger.info(logits)
