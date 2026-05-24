from __future__ import annotations

import argparse

import torch

from . import SpeculativeEngine, StandaloneDraft


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minisgl.speculative",
        description="Offline vanilla (greedy) speculative decoding.",
    )
    parser.add_argument(
        "--target-model",
        "--target",
        required=True,
        help="Target model path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--speculative-draft-model",
        "--draft-model",
        "--draft",
        default=None,
        help="Draft model path/repo id. Provide to ENABLE speculation; omit for plain greedy.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "-k",
        "--speculative-num-draft-tokens",
        dest="k",
        type=int,
        default=4,
        help="Drafts proposed per round (K).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.speculative_draft_model is None:
        _run_greedy(args)
    else:
        _run_speculative(args)


def _run_speculative(args: argparse.Namespace) -> None:
    draft = StandaloneDraft(args.speculative_draft_model, k=args.k, device=args.device)
    engine = SpeculativeEngine(args.target_model, draft=draft, k=args.k, device=args.device)
    output_ids = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(engine.tokenizer.decode(output_ids))
    print(
        f"\n[speculative  accept_length={engine.accept_length:.2f}  "
        f"accept_rate={engine.accept_rate:.2f}  target_forwards={engine.target_forward_ct}]"
    )


def _run_greedy(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.target_model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True))
    print("\n[greedy]")


if __name__ == "__main__":
    main()
