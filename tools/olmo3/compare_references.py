from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one OLMo3 HF/Mini reference")
    parser.add_argument("hf_reference", type=Path)
    parser.add_argument("mini_reference", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf = torch.load(args.hf_reference, map_location="cpu", weights_only=True)
    mini = torch.load(args.mini_reference, map_location="cpu", weights_only=True)
    assert torch.equal(hf["input_ids"].long(), mini["input_ids"].long())
    hf_logits = hf["first_logits"].float()
    mini_logits = mini["first_logits"].float()
    assert hf_logits.shape == mini_logits.shape
    assert torch.isfinite(hf_logits).all() and torch.isfinite(mini_logits).all()

    error = (hf_logits - mini_logits).abs()
    hf_top20 = set(hf_logits.topk(20).indices.tolist())
    mini_top20 = set(mini_logits.topk(20).indices.tolist())
    hf_top2 = hf_logits.topk(2).values
    metrics = {
        "argmax_match": hf_logits.argmax().item() == mini_logits.argmax().item(),
        "greedy_tokens_match": hf["token_ids"] == mini["token_ids"],
        "hf_token_ids": hf["token_ids"],
        "mini_token_ids": mini["token_ids"],
        "cosine_similarity": F.cosine_similarity(hf_logits, mini_logits, dim=0).item(),
        "mean_absolute_error": error.mean().item(),
        "max_absolute_error": error.max().item(),
        "top20_overlap": len(hf_top20 & mini_top20),
        "hf_top1_margin": (hf_top2[0] - hf_top2[1]).item(),
    }
    print(json.dumps(metrics, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metrics, indent=2) + "\n")

    assert metrics["argmax_match"]
    assert metrics["greedy_tokens_match"]
    assert metrics["cosine_similarity"] >= 0.999
    assert metrics["mean_absolute_error"] <= 0.05
    assert metrics["max_absolute_error"] <= 0.5
    assert metrics["top20_overlap"] >= 18
    print("OLMO3_ALIGNMENT=passed")


if __name__ == "__main__":
    main()
