from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.llm.llm import LLM
from minisgl.scheduler import Scheduler, SchedulerConfig


class TPReferenceLLM(LLM):
    def __init__(self, model_path: str, rank: int, world_size: int):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(rank, world_size),
            dtype=torch.bfloat16,
            offline_mode=True,
            attention_backend="fi",
            max_running_req=1,
            max_seq_len_override=64,
            max_extend_tokens=64,
            num_page_override=64,
            page_size=1,
            cache_type="naive",
            cuda_graph_bs=[],
        )
        Scheduler.__init__(self, config)
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0


def _worker(
    rank: int,
    world_size: int,
    model_path: str,
    hf_reference: str,
    output: str,
) -> None:
    reference = torch.load(hf_reference, map_location="cpu", weights_only=True)
    llm = TPReferenceLLM(model_path, rank, world_size)
    captured: list[torch.Tensor] = []
    sampled_tokens: list[int] = []
    original_sample = llm.engine.sampler.sample

    def capture_sample(logits, sampling_args):
        if not captured:
            captured.append(logits[0].detach().float().cpu().clone())
        next_tokens = original_sample(logits, sampling_args)
        sampled_tokens.extend(next_tokens.detach().cpu().tolist())
        return next_tokens

    llm.engine.sampler.sample = capture_sample
    try:
        llm.generate(
            [reference["input_ids"].tolist()],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
        )
    finally:
        llm.shutdown()

    assert len(captured) == 1
    assert len(sampled_tokens) == 4
    if rank == 0:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": reference["input_ids"],
                "first_logits": captured[0],
                "token_ids": sampled_tokens,
            },
            output_path,
        )
        print(f"TP_REFERENCE=ok token_ids={sampled_tokens} output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one minimal OLMo3 TP reference")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("hf_reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--world-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            str(args.model_path),
            str(args.hf_reference),
            str(args.output),
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
