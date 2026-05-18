"""Phase 2 integration test: DummyDraft must produce reference-greedy output.

A deliberately wrong draft exercises the rejection path. Output correctness is
the load-bearing assertion; the all-zero-acceptance assertion is the secondary
check that the dummy is actually doing its job.

Requires a CUDA GPU and the local Qwen3-1.7B weights at
``/home/javierlimt6/work/models/Qwen3-1.7B``. Skipped otherwise.
"""
from __future__ import annotations

import os

import pytest
import torch

from minisgl.speculative import DummyDraft, SpeculativeEngine

TARGET_PATH = "/home/javierlimt6/work/models/Qwen3-1.7B"
PROMPTS = [
    "The capital of France is",
    "def is_palindrome(s):",
    "Once upon a time,",
]
MAX_NEW = 100
K = 4

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.isdir(TARGET_PATH), reason=f"missing {TARGET_PATH}"),
]


@pytest.fixture(scope="module")
def spec_engine():
    engine = SpeculativeEngine(
        target_path=TARGET_PATH,
        draft=DummyDraft(k=K),
        k=K,
    )
    yield engine
    del engine
    torch.cuda.empty_cache()


def _reference_greedy(engine: SpeculativeEngine, prompt: str) -> list[int]:
    """Reference greedy via ``model.generate`` reusing the same model weights."""
    prompt_ids = engine.tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.int64, device=engine.device)
    with torch.inference_mode():
        out = engine.target.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            num_beams=1,
            pad_token_id=engine.tokenizer.eos_token_id,
        )
    new_tokens = out[0, len(prompt_ids):].tolist()
    eos = engine.tokenizer.eos_token_id
    if eos is not None and eos in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(eos)]
    return new_tokens


@pytest.mark.parametrize("prompt", PROMPTS)
def test_dummy_draft_matches_reference(spec_engine: SpeculativeEngine, prompt: str):
    spec_engine.reset_stats()
    spec_tokens = spec_engine.generate(prompt, max_new_tokens=MAX_NEW)
    ref_tokens = _reference_greedy(spec_engine, prompt)

    # Trim both to the shorter length; reference may stop on EOS earlier.
    n = min(len(spec_tokens), len(ref_tokens))
    assert n > 0, "both outputs are empty"
    assert spec_tokens[:n] == ref_tokens[:n], (
        f"divergence at index "
        f"{next((i for i in range(n) if spec_tokens[i] != ref_tokens[i]), n)}"
    )
    # Dummy always proposes a token that's effectively impossible as the
    # natural greedy continuation; every speculative round should accept zero.
    assert all(a == 0 for a in spec_engine.accepted_lengths), (
        f"unexpected acceptance: {spec_engine.accepted_lengths}"
    )
