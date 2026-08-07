from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one minimal OLMo3 HF reference")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = "Answer with one word: what is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    generated: list[int] = []
    first_logits: torch.Tensor | None = None
    current_ids = input_ids
    past_key_values = None
    with torch.inference_mode():
        for _ in range(4):
            output = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = output.logits[0, -1].float()
            if first_logits is None:
                first_logits = logits.cpu()
            next_token = logits.argmax().view(1, 1)
            generated.append(int(next_token.item()))
            current_ids = next_token
            past_key_values = output.past_key_values

    assert first_logits is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt": prompt,
            "input_ids": input_ids[0].cpu(),
            "first_logits": first_logits,
            "token_ids": generated,
        },
        args.output,
    )
    print(
        f"HF_REFERENCE=ok input_tokens={input_ids.shape[-1]} "
        f"token_ids={generated} output={args.output}"
    )


if __name__ == "__main__":
    main()
