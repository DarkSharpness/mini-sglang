from __future__ import annotations

import argparse

from . import SpeculativeEngine, StandaloneDraft


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minisgl.speculative",
        description="Offline vanilla (greedy) speculative decoding.",
    )
    parser.add_argument("--target-model", required=True, help="Target model path or HF repo id.")
    parser.add_argument("--draft-model", required=True, help="Draft model path or HF repo id.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("-k", type=int, default=4, help="Drafts proposed per round (default: 4).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    engine = SpeculativeEngine(
        args.target_model, draft=StandaloneDraft(args.draft_model, k=args.k), k=args.k
    )
    output_ids = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(engine.tokenizer.decode(output_ids, skip_special_tokens=True))
    print(f"\n[accept_length={engine.accept_length:.2f}  accept_rate={engine.accept_rate:.2f}]")


if __name__ == "__main__":
    main()
