# Attention Backends

The `attention/` component provides pluggable attention kernels. All backends implement the same `BaseAttnBackend` interface so the rest of the system is kernel-agnostic.

---

## Backend Hierarchy

```
BaseAttnBackend  (abstract)
    │
    ├── FlashAttentionBackend   (fa)     ← sgl_kernel flash_attn_with_kvcache
    ├── FlashInferBackend       (fi)     ← flashinfer BatchPrefill/Decode wrappers
    ├── TRTLLMBackend          (trtllm)  ← TensorRT-LLM paged attention
    └── HybridBackend                   ← delegates to prefill_backend / decode_backend
```

**Auto-selection** at startup (when `attention_backend = "auto"`):
```
SM 10.0+ (Blackwell) → "trtllm"
SM 9.0+  (Hopper)    → "fa,fi"   (FA for prefill, FlashInfer for decode)
otherwise            → "fi"      (FlashInfer only)
```

`"fa,fi"` creates a `HybridBackend(prefill=FA, decode=FI)`.

---

## BaseAttnBackend Interface

```python
forward(q, k, v, layer_id, batch) → Tensor   # compute attention + store KV
prepare_metadata(batch)                        # build kernel-specific metadata before forward
init_capture_graph(max_seq_len, bs_list)       # allocate static buffers for CUDA graph capture
prepare_for_capture(batch)                     # set up a specific bs for graph capture
prepare_for_replay(batch)                      # update static buffers before graph replay
```

---

## Data Flow per Layer

```
AttentionLayer.forward(qkv)
    │
    ├── split qkv → q, k, v
    ├── (optional) q_norm, k_norm
    ├── rotary embedding
    │
    ▼
ctx.attn_backend.forward(q, k, v, layer_id, batch)
    │
    ├── store_kv(k, v, batch.out_loc, layer_id)   ← scatter K/V into paged cache
    └── run kernel(q, k_cache, v_cache, metadata) → output
```

`batch.out_loc` is pre-computed by the scheduler: physical token addresses in the KV cache where this layer's K/V should be written.

---

## FlashAttentionBackend (FA)

Uses `sgl_kernel.flash_attn.flash_attn_with_kvcache`.

**Metadata** (`FAMetadata`):
```
cu_seqlens_q   [bs+1]   cumulative query sequence lengths
cu_seqlens_k   [bs+1]   cumulative key sequence lengths (including cached)
cache_seqlens  [bs]     total K length per request
page_table     [bs, max_pages]  page indices (divided by page_size)
max_seqlen_q / max_seqlen_k
```

Supports paged KV via page_table. FA version 3 (Hopper) or 4 (Blackwell) selected automatically.

---

## FlashInferBackend (FI)

Uses `flashinfer` `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper`.

**Metadata** (`FIMetadata`):
```
cu_seqlens_q/k  (CPU + GPU copies)
indices         [total_tokens]  flat list of physical page addresses (page_size=1)
last_page_len   [bs]            always 1 (since page_size=1)
wrapper         ← prefill or decode wrapper, chosen per batch.is_prefill
```

FlashInfer requires `page_size=1` — the KV cache is treated as a flat token array.

`_initialize_metadata_once`: FlashInfer's `.plan()` must be called once per metadata object before `.run()`. A CUDA event serializes plan calls to avoid buffer races.

**Tensor cores** for decode: enabled when GQA ratio ≥ 4 (i.e., `num_qo_heads / num_kv_heads ≥ 4`).

---

## CUDA Graph Integration

Each backend has a two-phase graph protocol:

```
Capture phase (startup):
    init_capture_graph(max_seq_len, bs_list)   ← allocate static CUDAGraph buffers
    for each bs:
        prepare_for_capture(batch)              ← point metadata at static buffers
        torch.cuda.graph(...)
            model.forward()                     ← recorded

Replay phase (runtime):
    prepare_for_replay(batch)                   ← copy live metadata into static buffers
    graph.replay()
```

`HybridBackend` delegates `init_capture_graph`, `prepare_for_capture`, `prepare_for_replay` to `decode_backend` only (prefill is never graph-captured).

---

## AttnMetadata.get_last_indices

Used by the sampler to extract the logit for the last token of each request:

```
FA:  cu_seqlens_q[1:bs+1] - 1
FI:  cu_seqlens_q_gpu[1:bs+1] - 1
```

This gives the index of the last query token position within the flattened token batch.

---

## Key Files

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseAttnBackend`, `BaseAttnMetadata`, `HybridBackend` |
| `fa.py` | `FlashAttentionBackend`, `FAMetadata` |
| `fi.py` | `FlashInferBackend`, `FIMetadata` |
| `trtllm.py` | `TRTLLMBackend` |
| `utils.py` | `BaseCaptureData` shared between backends |
