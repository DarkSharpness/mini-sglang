"""Diagnostic probe for the Phase 2 dummy-draft divergence at token 8.

Hypothesis A: DynamicCache.crop() leaves stale KV entries that pollute reads.
Hypothesis B: bfloat16 near-ties produce argmax instability between full-prefix
              and incremental forwards.

Strategy: replicate the speculative loop manually, and at every accepted-token
position compute the *one-shot reference logits* by running a fresh forward
over the full committed prefix. If the speculative logits and the fresh logits
match exactly at every step, the engine is correct and the failure is upstream
of the engine. If they diverge, we know exactly where and on which value.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from minisgl.speculative.verify import verify_drafts

TARGET_PATH = "/home/javierlimt6/work/models/Qwen3-1.7B"
PROMPT = "Once upon a time,"
MAX_NEW = 12  # well past index 8
K = 4
WRONG = 100_000


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TARGET_PATH)
    model = AutoModelForCausalLM.from_pretrained(TARGET_PATH, dtype=torch.bfloat16).to(
        "cuda"
    ).eval()
    prompt_ids = tok.encode(PROMPT)
    print(f"prompt_ids ({len(prompt_ids)}):", prompt_ids)

    # 1) Reference greedy (per-token).
    with torch.inference_mode():
        ref_ids = model.generate(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            max_new_tokens=MAX_NEW,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )[0].tolist()
    ref_new = ref_ids[len(prompt_ids):]
    print(f"reference new tokens ({len(ref_new)}):", ref_new)

    # 2) Speculative loop (replicated inline for instrumentation).
    spec_new: list[int] = []
    cache = DynamicCache()
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        t_last = int(out.logits[0, -1].argmax().item())
        spec_new.append(t_last)

        # Sanity: the first generated token must equal ref_new[0].
        assert t_last == ref_new[0], f"prefill divergence: {t_last} vs {ref_new[0]}"

        while len(spec_new) < MAX_NEW:
            drafts = [WRONG] * K
            verify_input = torch.tensor(
                [[t_last, *drafts]], dtype=torch.int64, device="cuda"
            )
            cache_len_before = cache.get_seq_length()
            out = model(
                input_ids=verify_input,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values

            verify_logits = out.logits[0]  # (K+1, vocab)

            # Diagnostic: also compute logits from a fresh forward over the
            # full committed prefix [prompt + spec_new]. The logit at the
            # last position of that forward should match verify_logits[0].
            full_prefix = prompt_ids + spec_new
            fresh_out = model(
                input_ids=torch.tensor([full_prefix], device="cuda"),
                use_cache=False,
            )
            fresh_last_logit = fresh_out.logits[0, -1]  # (vocab,)

            spec_pos0_argmax = int(verify_logits[0].argmax().item())
            fresh_argmax = int(fresh_last_logit.argmax().item())
            top5_spec = verify_logits[0].topk(5)
            top5_fresh = fresh_last_logit.topk(5)
            print(
                f"step {len(spec_new):2d}  cache_before={cache_len_before:3d}  "
                f"spec_argmax={spec_pos0_argmax:6d}  fresh_argmax={fresh_argmax:6d}  "
                f"match={spec_pos0_argmax == fresh_argmax}"
            )
            if spec_pos0_argmax != fresh_argmax:
                print(f"  spec  top5 ids: {top5_spec.indices.tolist()}")
                print(f"  spec  top5 logits: {[f'{v:.4f}' for v in top5_spec.values.tolist()]}")
                print(f"  fresh top5 ids: {top5_fresh.indices.tolist()}")
                print(f"  fresh top5 logits: {[f'{v:.4f}' for v in top5_fresh.values.tolist()]}")
                print(f"  max abs diff: "
                      f"{(verify_logits[0].float() - fresh_last_logit.float()).abs().max().item():.6f}")

            drafts_t = torch.tensor(drafts, dtype=torch.int64, device="cuda")
            accepted_tokens, accepted_drafts = verify_drafts(drafts_t, verify_logits)
            rejected = K - accepted_drafts
            cache.crop(cache.get_seq_length() - rejected)
            spec_new.extend(accepted_tokens.tolist())
            t_last = spec_new[-1]

    print(f"\nspec new tokens: {spec_new[:MAX_NEW]}")
    print(f"ref  new tokens: {ref_new[:MAX_NEW]}")
    for i in range(min(len(spec_new), len(ref_new), MAX_NEW)):
        if spec_new[i] != ref_new[i]:
            print(f"diverge at index {i}: spec={spec_new[i]} ref={ref_new[i]}")
            break
    else:
        print("all match!")


if __name__ == "__main__":
    main()
