# KV Cache

The `kvcache/` component provides two orthogonal abstractions:

1. **KV Cache Pool** — GPU memory holding the actual key/value tensors, indexed by physical page.
2. **Prefix Cache** — a logical index mapping token sequences to physical page addresses, enabling prompt-prefix reuse across requests.

---

## Abstraction Layers

```
┌───────────────────────────────────────────────────────────┐
│                    Scheduler / Engine                      │
└────────────────────────────┬──────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   BasePrefixCache                   BaseKVCachePool
   (logical index)                   (physical GPU memory)
        │                                    │
   ┌────┴─────┐                     ┌────────┴────────┐
   │  Radix   │                     │   MHAKVCache    │
   │  Cache   │                     │ (paged tensors) │
   └──────────┘                     └─────────────────┘
   NaiveCache
   (no reuse)
```

---

## BaseKVCachePool — Physical Storage

`MHAKVCache` allocates a single contiguous tensor for all layers:

```
_kv_buffer: Tensor[2, num_layers, num_pages+1, page_size, local_kv_heads, head_dim]
             │
             ├── [0] → K buffer   (k_cache)
             └── [1] → V buffer   (v_cache)

              num_pages+1 includes one "dummy page" (index num_pages)
              pointed to by dummy requests during CUDA graph capture.
```

**store_kv(k, v, out_loc, layer_id)**

```
out_loc: Tensor[total_tokens]  ← physical token addresses
         │
         ▼
k_cache[layer_id].view(num_pages*page_size, heads, dim)[out_loc] ← k
v_cache[layer_id].view(num_pages*page_size, heads, dim)[out_loc] ← v
```

Writing uses the `store_cache` custom kernel for coalesced scatter writes.

---

## BasePrefixCache — Logical Index

### RadixPrefixCache

A radix tree (compressed trie) keyed on page-aligned token sequences. Each tree node stores a slice of token IDs and their corresponding physical page addresses.

```
root (always protected, ref_count=1)
  ├── [tok0..tokN]  →  node_A  (pages [p0..pN])
  │       └── [tokN+1..tokM]  →  node_B  (pages [pN+1..pM])
  └── [tok0..tokK]  →  node_C  ...

ref_count > 0  → "protected" (in use by an active request, cannot evict)
ref_count = 0  → "evictable" (LRU candidate)
```

**match_prefix(input_ids)**
- Walk tree greedily, aligning matches to `page_size` boundaries.
- Split a node if only a prefix of it matches.
- Returns a `RadixCacheHandle(cached_len, node)`.

**insert_prefix(input_ids, indices)**
- Walk to deepest match, then append a new node for the unmatched suffix (page-aligned).
- Returns how many tokens were already in cache before insertion.

**evict(size)**
- Collect leaf nodes with `ref_count == 0`, heap-sort by LRU timestamp.
- Pop leaves until enough pages freed; orphaned parents that become leaves are added to heap.

### NaiveCache

No prefix reuse. Every `match_prefix` returns 0 cached tokens. Useful for benchmarking or correctness testing.

---

## Handle & Lock Protocol

```
match_prefix(input_ids)  →  MatchResult(cuda_handle)
        │
        │  lock_handle(handle)   ← protect from eviction during scheduling
        │
        ▼
  req begins prefill
        │
        │  insert_prefix(...)    ← add new KV content to cache
        │  unlock_handle(handle) ← allow eviction of old handle
        ▼
  req finishes → cache_req(req, finished=True)
                  └── free tail pages that couldn't be inserted
```

Locking increments `ref_count` on all ancestor nodes; unlocking decrements. A node is evictable only when `ref_count == 0`.

---

## Size Accounting

```
SizeInfo(evictable_size, protected_size)
    total_size = evictable_size + protected_size   (in tokens)

CacheManager.available_size
    = prefix_cache.evictable_size
    + len(free_slots) * page_size
```

The CacheManager merges free physical slots with evictable prefix cache pages into a single "available" budget used by the prefill adder.

---

## Key Files

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseKVCachePool`, `BasePrefixCache`, `BaseCacheHandle`, `SizeInfo` |
| `mha_pool.py` | `MHAKVCache` — paged GPU tensor storage |
| `radix_cache.py` | `RadixPrefixCache`, `RadixTreeNode`, LRU eviction |
| `naive_cache.py` | `NaivePrefixCache` — no-op prefix matching |
