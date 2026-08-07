from __future__ import annotations

import argparse
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one minimal OLMo3 scheduler smoke")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("hf_reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = torch.load(args.hf_reference, map_location="cpu", weights_only=True)
    input_ids = reference["input_ids"].tolist()
    expected_token = reference["token_ids"][0]
    llm = LLM(
        str(args.model_path),
        attention_backend="fi",
        max_running_req=2,
        max_seq_len_override=64,
        max_extend_tokens=64,
        num_page_override=256,
        page_size=1,
        cache_type="radix",
        cuda_graph_bs=[],
    )

    prefill_calls = 0
    max_prefill_batch = 0
    matched_lengths: list[int] = []
    original_forward = llm.engine.forward_batch
    original_match = llm.cache_manager.match_req

    def capture_forward(batch, sampling_args):
        nonlocal prefill_calls, max_prefill_batch
        if batch.is_prefill:
            prefill_calls += 1
            max_prefill_batch = max(max_prefill_batch, batch.size)
        return original_forward(batch, sampling_args)

    def capture_match(req):
        result = original_match(req)
        matched_lengths.append(result.cuda_handle.cached_len)
        return result

    llm.engine.forward_batch = capture_forward
    llm.cache_manager.match_req = capture_match
    sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    try:
        batched = llm.generate([input_ids, input_ids], sampling)
        first_pass_matches = list(matched_lengths)
        matched_lengths.clear()
        cached = llm.generate([input_ids], sampling)
        second_pass_matches = list(matched_lengths)
        llm.cache_manager.check_integrity()
    finally:
        llm.shutdown()

    assert [item["token_ids"] for item in batched] == [
        [expected_token],
        [expected_token],
    ]
    assert cached[0]["token_ids"] == [expected_token]
    assert max_prefill_batch == 2
    assert prefill_calls >= 2
    assert first_pass_matches == [0, 0]
    assert second_pass_matches and second_pass_matches[0] >= len(input_ids) - 1
    print(
        "OLMO3_SYSTEM_SMOKE=passed "
        f"prefill_calls={prefill_calls} max_prefill_batch={max_prefill_batch} "
        f"radix_cached_tokens={second_pass_matches[0]}"
    )


if __name__ == "__main__":
    main()
