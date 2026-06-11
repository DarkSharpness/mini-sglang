# Engine

The `engine/` component is the GPU-side execution core. It owns the model weights, KV cache allocation, CUDA graph replay, and token sampling. The scheduler calls into it once per batch.

---

## Component Map

```
EngineConfig
    │
    ▼
Engine
  ├── model (BaseLLMModel)          ← loaded from HuggingFace weights
  ├── kv_cache (BaseKVCachePool)    ← paged GPU memory for K/V tensors
  ├── page_table (int32 tensor)     ← [max_req+1, max_seq_len] physical location lookup
  ├── attn_backend                  ← FlashAttention / FlashInfer / TRTLLM
  ├── moe_backend                   ← (optional) fused MoE kernel
  ├── sampler (Sampler)             ← greedy / top-k / top-p token selection
  └── graph_runner (GraphRunner)    ← CUDA graph capture + replay for decode
```

---

## Initialization Sequence

```
1. Set CUDA device, random seed, CUDA stream
2. Init distributed (gloo + optional pynccl)
3. Measure free GPU memory  ─────────────────────────────┐
4. Load model weights onto GPU                           │ used to compute
5. Measure free GPU memory again  ───────────────────────┘ KV cache budget
6. Allocate KV cache pages  (memory_ratio * init_free − model_memory)
7. Allocate page table tensor
8. Create attention backend
9. Create MoE backend (if MoE model)
10. Create Sampler
11. Capture CUDA graphs (GraphRunner)
```

---

## `forward_batch` Call Flow

```
Scheduler calls forward_batch(batch, args)
        │
        ▼
  ctx.forward_batch(batch)      ← sets global context so layers can read batch
        │
        ├─ can_use_cuda_graph?
        │       ├── YES → graph_runner.replay(batch)   (decode, small bs)
        │       └── NO  → model.forward()              (prefill or large bs)
        │
        ▼
  req.complete_one() for each req   ← advances cached_len / device_len
        │
        ▼
  sampler.sample(logits, args)   → next_tokens_gpu
        │
        ▼
  async D2H copy of next_tokens_cpu
        │
        ▼
  ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

---

## Memory Budget Calculation

```
init_free_memory  (before weight load)
model_memory      = init_free - post_load_free
available_memory  = memory_ratio × init_free - model_memory
num_pages         = available_memory ÷ cache_per_page

cache_per_page = 2 (K+V) × head_dim × local_kv_heads × page_size × dtype_bytes × num_layers
```

All TP ranks synchronize their free-memory values via CPU all-reduce and take the minimum so KV cache sizes are identical across ranks.

---

## GraphRunner (CUDA Graph Capture)

CUDA graphs pre-record the full decode forward pass for each supported batch size, eliminating CPU kernel-launch overhead during decode.

```
Startup (capture phase)
─────────────────────
For each bs in [1, 2, 4, 8, 16, ..., max_bs]:
    create dummy Batch(reqs=[dummy_req] * bs)
    warmup run  (not recorded)
    torch.cuda.graph(...)
      └── model.forward()           ← recorded into CUDAGraph
    graph_map[bs] = graph

Runtime (replay phase)
──────────────────────
batch arrives (decode, bs ≤ max_graph_bs)
    copy batch tensors → GraphCaptureBuffer (pre-allocated static buffers)
    attn_backend.prepare_for_replay(batch)  ← update paged KV metadata
    graph_map[padded_bs].replay()
    return buffer.logits[:real_bs]
```

Batch sizes are padded to the next captured size using a dummy request that points to a dummy KV cache page, so no out-of-bounds reads occur.

---

## Sampler

```
BatchSamplingArgs (per-batch, GPU)
    ├── temperatures   [bs]
    ├── top_k          [bs]
    └── top_p          [bs]

sampler.sample(logits, args)
    ├── temperature scaling
    ├── top-k filtering
    ├── top-p (nucleus) filtering
    └── multinomial or argmax → token ids [bs]
```

---

## Key Files

| File | Responsibility |
|------|---------------|
| `engine.py` | Engine class, init, `forward_batch` |
| `config.py` | `EngineConfig` dataclass |
| `graph.py` | `GraphRunner`, CUDA graph capture/replay |
| `sample.py` | `Sampler`, `BatchSamplingArgs` |
