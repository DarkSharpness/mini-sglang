from __future__ import annotations

import argparse
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one minimal OLMo3 Mini reference")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("hf_reference", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = torch.load(args.hf_reference, map_location="cpu", weights_only=True)
    llm = LLM(
        str(args.model_path),
        attention_backend="fi",
        max_running_req=1,
        max_seq_len_override=64,
        max_extend_tokens=64,
        num_page_override=64,
        page_size=1,
        cache_type="naive",
        cuda_graph_bs=[],
    )
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
        result = llm.generate(
            [reference["input_ids"].tolist()],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
        )[0]
    finally:
        llm.shutdown()

    assert len(captured) == 1
    assert len(sampled_tokens) == 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": reference["input_ids"],
            "first_logits": captured[0],
            "token_ids": sampled_tokens,
        },
        args.output,
    )
    print(
        f"MINI_REFERENCE=ok token_ids={sampled_tokens} "
        f"decoded_token_ids={result['token_ids']} output={args.output}"
    )


if __name__ == "__main__":
    main()
