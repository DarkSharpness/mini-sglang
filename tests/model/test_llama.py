from __future__ import annotations
from typing import Any

import torch

from transformers import AutoConfig
from minisgl.attention import create_attention_backend
from minisgl.config.context import Batch, Context, Req, set_global_ctx
from minisgl.config.model import ModelConfig
from minisgl.distributed import set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.models.llama import LlamaForCausalLM
from minisgl.utils import call_if_main
from minisgl.utils.torch_utils import torch_dtype


@call_if_main()
def main():
    set_tp_info(0, 1)
    config: Any = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_config = ModelConfig.from_hf(config)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    with torch_dtype(dtype), device:
        model = LlamaForCausalLM(model_config)

    kv_cache = create_kvcache(
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_pages=65536,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )

    attn_backend = create_attention_backend(kv_cache, "fa3")

    ctx = Context(
        page_num=65536,
        page_size=1,
        max_running_req=256,
        max_seq_len=131072,
        device=device,
        kv_cache=kv_cache,
        attn_backend=attn_backend,
    )
    set_global_ctx(ctx)

    batch = Batch(
        reqs=[
            Req(
                input_ids=[0] * 1024,
                page_table_idx=1,
                cached_len=0,
                output_len=10,
                device=device,
                rid=0,
            )
        ],
    )

    attn_backend.prepare_metadata(batch, allow_graph=False)
    batch.attn_metadata.finalize(ctx.page_table)
    logits = model.forward_batch(batch)
    print(logits.shape)
