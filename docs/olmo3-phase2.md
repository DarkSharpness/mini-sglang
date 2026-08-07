# OLMo3 phase 2: official-weight alignment

Phase 2 validates the phase-1 `TP=1` implementation against the official BF16
checkpoint. It is a correctness check, not a performance benchmark.

## Fixed inputs

- Model: `allenai/Olmo-3-7B-Instruct`
- Snapshot revision: `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`
- Local snapshot: `/root/autodl-tmp/models/Olmo-3-7B-Instruct`
- Prompt: `Answer with one word: what is the capital of France?`
- Chat-template input length: 48 tokens
- Precision: BF16
- GPU: one RTX 4090

The three safetensors files have 355 tensors and 7,298,011,136 parameters. Their
SHA-256 values match the Hugging Face LFS metadata, every tensor is BF16, and the
index has no missing or extra keys.

## Procedure

HF Transformers and Mini-SGLang run in separate processes so that the two 7B
models never occupy the GPU together. Both use the same local snapshot and run
offline. CUDA graphs and overlap scheduling are disabled for this one comparison.

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1

python tools/olmo3/hf_reference.py \
  /root/autodl-tmp/models/Olmo-3-7B-Instruct \
  /root/autodl-tmp/olmo3-validation/hf_reference.pt

python tools/olmo3/mini_reference.py \
  /root/autodl-tmp/models/Olmo-3-7B-Instruct \
  /root/autodl-tmp/olmo3-validation/hf_reference.pt \
  /root/autodl-tmp/olmo3-validation/mini_reference.pt

python tools/olmo3/compare_references.py \
  /root/autodl-tmp/olmo3-validation/hf_reference.pt \
  /root/autodl-tmp/olmo3-validation/mini_reference.pt \
  --json-output /root/autodl-tmp/olmo3-validation/metrics.json
```

Only the first prefill logits and four greedy tokens are retained. The result
files are small and no latency or throughput measurements are collected.

## Result

```json
{
  "argmax_match": true,
  "greedy_tokens_match": true,
  "hf_token_ids": [60704, 100257, 100264, 78191],
  "mini_token_ids": [60704, 100257, 100264, 78191],
  "cosine_similarity": 0.9999091029167175,
  "mean_absolute_error": 0.024406513199210167,
  "max_absolute_error": 0.125,
  "top20_overlap": 20,
  "hf_top1_margin": 5.625
}
```

All acceptance thresholds passed: identical argmax and four-token sequence,
cosine similarity at least 0.999, mean absolute error at most 0.05, maximum
absolute error at most 0.5, and top-20 overlap at least 18.

## Additional fix

The official generation configuration has two EOS tokens: `<|endoftext|>`
(`100257`) and `<|im_end|>` (`100265`). Mini-SGLang now loads the complete EOS
set and uses it consistently in the scheduler, offline LLM output, and online
detokenizer instead of recognizing only the tokenizer's primary EOS token.

## Remaining scope

- OLMo3 `TP=2` remains intentionally unsupported until projection-wide Q/K
  RMSNorm has correct cross-rank statistics.
- Sliding layers retain full KV storage; physical window eviction is not part of
  this phase.
- The FlashInfer path is the validated RTX 4090 backend. FlashAttention remains
  blocked by the current `sgl_kernel`/CUTLASS environment.
- No benchmark matrix was run.
