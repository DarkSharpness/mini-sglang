# OLMo3 phase 3: tensor parallel correctness

Phase 3 adds correct OLMo3 tensor parallel execution for the official 7B model
and validates `TP=2` against the phase-2 HF and Mini-SGLang `TP=1` references.
It does not include performance benchmarking.

## Implementation

OLMo3 applies Q/K RMSNorm across each complete projection before reshaping into
heads. With tensor parallelism, every rank owns only a contiguous projection
shard. `DistributedRMSNorm` therefore computes a local FP32 square sum, performs
a cross-rank SUM all-reduce, divides by the full projection width, and applies
the corresponding local weight shard. Q/K activations are never all-gathered.

The official OLMo3 7B projections are 4096 elements wide. At `TP=2`, each rank
holds 2048 Q-Norm and K-Norm weights. The loader shards only OLMo3 keys matching
`self_attn.q_norm.weight` or `self_attn.k_norm.weight`; Qwen3's replicated
per-head norms remain unchanged.

Existing Mini-SGLang TP behavior is reused for QKV, O projection, MLP,
embedding, and LM head. Models whose KV head count is smaller than TP size are
explicitly rejected until replicated-KV normalization is implemented.

PyNCCL does not currently support the FP32 distributed statistics required by
this normalization. OLMo3 with `TP>1` therefore automatically selects standard
`torch.distributed` NCCL, which is the validated communication path.

## Validation environment

- Two RTX 4090 GPUs connected through PCIe PXB, without NVLink or P2P access.
- `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` force the supported fallback.
- FlashInfer attention backend.
- BF16 official snapshot revision
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`.
- One 48-token input and four greedy output tokens.
- CUDA graphs and overlap scheduling disabled for the comparison.

The reproducible entry point is `tools/olmo3/tp_reference.py`. It launches two
offline scheduler processes, binds one process to each GPU, and saves rank 0's
first prefill logits and four sampled tokens.

## Result

All HF, `TP=1`, and `TP=2` greedy token IDs are identical:

```text
[60704, 100257, 100264, 78191]
```

HF versus `TP=2`:

```json
{
  "cosine_similarity": 0.9999080300331116,
  "mean_absolute_error": 0.023565489798784256,
  "max_absolute_error": 0.21875,
  "top20_overlap": 20,
  "argmax_match": true,
  "greedy_tokens_match": true
}
```

Mini-SGLang `TP=1` versus `TP=2`:

```json
{
  "cosine_similarity": 0.9998343586921692,
  "mean_absolute_error": 0.03872865438461304,
  "max_absolute_error": 0.1875,
  "top20_overlap": 20,
  "argmax_match": true,
  "greedy_tokens_match": true
}
```

Both comparisons pass the fixed correctness thresholds. Result files are stored
under `/root/autodl-tmp/olmo3-validation/` as `tp2_reference.pt`,
`tp2-vs-hf.json`, and `tp2-vs-tp1.json`.

## Remaining limitations

- Standard NCCL is the validated TP2 communication path; PyNCCL is automatically
  disabled for OLMo3 TP until it supports FP32 statistics.
- OLMo3 configurations requiring replicated KV heads remain unsupported.
- Sliding layers retain full KV storage.
- FlashAttention remains blocked by the current `sgl_kernel`/CUTLASS setup;
  FlashInfer is the validated RTX 4090 backend.
- No throughput, latency, concurrency, or long-context benchmark was run.
