# Scheduler

The `scheduler/` component is the CPU-side brain of the inference loop. It runs in its own process per TP rank, orchestrates prefill/decode batching, manages KV cache page allocation, and drives the engine with prepared batches.

---

## Component Map

```
Scheduler
  ├── engine (Engine)                 ← GPU execution core
  ├── table_manager (TableManager)    ← allocates request "slots" (table indices)
  ├── cache_manager (CacheManager)    ← paged KV memory + prefix cache eviction
  ├── prefill_manager (PrefillManager)← queue of pending / chunked prefill reqs
  └── decode_manager (DecodeManager)  ← set of in-flight decode reqs
```

---

## Main Loop: Overlap Scheduling

The scheduler overlaps CPU metadata work for the *next* batch with GPU execution of the *current* batch.

```
              ┌─────────────┐        ┌─────────────┐
  CPU thread  │ receive msgs│        │ process last│
              │ schedule_   │        │ batch output│
              │ next_batch  │        │ (tokens,    │
              │ prepare_    │        │  detokenize,│
              │ batch       │        │  free pages)│
              └──────┬──────┘        └──────┬──────┘
                     │ launch                │
              ┌──────▼──────────────────────▼──────┐
  GPU stream  │          forward_batch(current)     │
              └─────────────────────────────────────┘

Iteration N:   schedule N  →  launch N  →  process N-1 result
Iteration N+1: schedule N+1 →  launch N+1 →  process N result
```

A separate CUDA stream is used for CPU-side scheduling metadata; engine runs on its own stream. `engine_stream.wait_stream(scheduler_stream)` ensures ordering.

---

## Scheduling Priority

```
_schedule_next_batch()
    │
    ├── prefill_manager.schedule_next_batch(prefill_budget)   ← FIRST priority
    │       └── returns None if no pending reqs
    │
    └── decode_manager.schedule_next_batch()                  ← SECOND priority
            └── returns None if no in-flight decode reqs
```

Prefill-first policy: ensures new requests make progress and enter the decode queue promptly.

---

## Prefill Manager

```
pending_list: [PendingReq, ...]   ← FIFO queue of new requests

schedule_next_batch(prefill_budget):
    PrefillAdder (token_budget=prefill_budget, reserved_size=decode_inflight_tokens)
    for each pending_req:
        ├── allocate table slot
        ├── match prefix cache  →  cached_len (may skip tokens)
        ├── chunk if extend_len > token_budget  →  ChunkedReq
        └── add to batch

    Chunked reqs stay at the front of pending_list for next iteration.
```

**Chunked prefill**: if a request's input is too long to fit the token budget in one step, it is split across multiple prefill batches. `ChunkedReq` marks that the req must not be sampled yet.

---

## Decode Manager

```
running_reqs: Set[Req]

filter_reqs(reqs):            ← called after each forward; adds newly-promoted reqs
    running_reqs = {r for r in running_reqs ∪ reqs if r.can_decode}

schedule_next_batch():
    Batch(reqs=sorted(running_reqs, key=uid), phase="decode")
```

Sorting by UID ensures stable ordering across TP ranks (critical for consistent sampling).

---

## Cache Manager

Sits between the prefix cache and the page table. Manages a free-slot pool of page-aligned slots.

```
free_slots: Tensor[int32]      ← available page start addresses

allocate_paged(reqs):
    needed_pages ← sum of new pages each req needs
    if needed_pages > free_slots:
        evict from prefix_cache → reclaim pages
    write newly allocated page addresses into page_table

cache_req(req, finished):
    insert req's input_ids + page indices → prefix_cache
    free previously-matched pages (they're now in cache)
    if finished: free the tail pages too

lazy_free_region():            ← context manager
    defers page frees until the GPU work using those pages completes
```

---

## Batch Preparation Pipeline

```
_prepare_batch(batch)
    │
    ├── graph_runner.pad_batch(batch)       ← pad to next CUDA graph size
    ├── cache_manager.allocate_paged(reqs)  ← write new pages to page_table
    ├── _make_positions(batch)              ← [total_tokens] position indices
    ├── _make_input_tuple(batch)            ← (table_idx, positions) for token lookup
    ├── _make_write_tuple(batch)            ← (table_idx, seq_len/-1) for KV write
    ├── batch.out_loc ← page_table[input_mapping]
    └── attn_backend.prepare_metadata(batch)

Returns ForwardInput(batch, sample_args, input_tuple, write_tuple)
```

---

## Message Handling

```
receive_msg(blocking)
    │
    ├── UserMsg       → prefill_manager.add_one_req(msg)
    ├── AbortBackendMsg → remove from prefill or decode manager, free resources
    ├── ExitMsg       → raise KeyboardInterrupt
    └── BatchBackendMsg → unwrap and process each sub-message
```

After each forward pass, `_process_last_data` iterates tokens, sends `DetokenizeMsg` replies, and frees finished request resources (table slot, cache pages).

---

## Key Files

| File | Responsibility |
|------|---------------|
| `scheduler.py` | Main loop, overlap scheduling, message dispatch |
| `prefill.py` | `PrefillManager`, `PrefillAdder`, chunked prefill |
| `decode.py` | `DecodeManager`, decode batch scheduling |
| `cache.py` | `CacheManager`, page allocation/eviction |
| `table.py` | `TableManager`, request slot allocation |
| `io.py` | ZMQ I/O mixin (send/receive messages) |
| `config.py` | `SchedulerConfig` dataclass |
| `utils.py` | `PendingReq` helper |
