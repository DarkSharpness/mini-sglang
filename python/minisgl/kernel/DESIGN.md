# Kernel

The `kernel/` component provides low-level GPU kernels and C++ extensions used throughout mini-sglang. It includes custom CUDA kernels compiled via PyTorch JIT, Triton kernels, and a Python-level NCCL wrapper.

---

## Component Map

```
kernel/
├── Python wrappers
│   ├── index.py       ← fast_index_put, fast_compare_key
│   ├── store.py       ← store_cache  (scatter K/V into paged memory)
│   ├── radix.py       ← fast_compare_key  (used by RadixTreeNode)
│   ├── tensor.py      ← Tensor C++ extension (serialization)
│   ├── pynccl.py      ← PyNCCL communicator wrapper
│   └── utils.py       ← fused_moe_kernel_triton, moe_sum_reduce_triton
│
├── Triton kernel
│   └── triton/fused_moe.py   ← Triton MoE matmul kernel
│
└── CUDA C++ sources (csrc/)
    ├── src/pynccl.cu     ← custom NCCL all_reduce / all_gather
    ├── src/radix.cpp     ← fast byte-level key compare
    ├── src/tensor.cpp    ← tensor serialization for IPC
    ├── jit/index.cu      ← scatter-index kernels
    └── jit/store.cu      ← paged KV store kernel
```

---

## store_cache — KV Scatter Kernel

```
store_cache(k_cache, v_cache, indices, k, v)

k_cache: [total_slots, heads, head_dim]   ← flattened paged KV tensor
v_cache: [total_slots, heads, head_dim]
indices: [num_tokens]                      ← physical slot addresses
k, v:   [num_tokens, heads, head_dim]     ← new KV from forward pass

Operation: k_cache[indices] = k
           v_cache[indices] = v

Implemented as a CUDA scatter write (index.cu / store.cu).
```

This is called once per transformer layer during forward, writing only the newly computed tokens (not the cached prefix).

---

## fast_compare_key — Radix Tree Key Comparison

```
fast_compare_key(node_key: Tensor, input_ids: Tensor) → int

Returns the length of the common prefix between node_key and input_ids.
Used by RadixTreeNode.get_match_len() during prefix cache tree walk.

Implemented in C++ (radix.cpp) for speed; avoids Python loop overhead
on potentially long token sequences.
```

---

## PyNCCL — Custom NCCL Wrapper

```
init_pynccl(tp_rank, tp_size, tp_cpu_group, max_size_bytes)
    → PyNCCLCommunicator

PyNCCLCommunicator:
    .all_reduce(tensor, op="sum")
    .all_gather(output, input)
```

PyNCCL avoids the PyTorch distributed NCCL overhead (Python GIL, extra dispatching) by calling NCCL operations directly from C++ via CUDA IPC. Particularly beneficial for small all-reduce tensors in tensor parallelism (e.g., hidden states after each linear layer).

A single pre-allocated GPU buffer of size `max_size_bytes` is reused for all NCCL operations to avoid dynamic allocation.

---

## Triton MoE Kernel

```
fused_moe_kernel_triton(
    hidden_states,    w,
    output,
    topk_weights,     topk_ids,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    apply_router_weight,
    topk,             config,
    compute_type
)
```

A Triton kernel that performs a batched matrix multiplication across multiple experts simultaneously. Tokens are sorted by expert assignment, padded to `BLOCK_SIZE_M`, and processed in tiled groups. Block sizes are tuned via `config` (see `moe/DESIGN.md`).

```
moe_sum_reduce_triton(intermediate_cache3, out_hidden_states)
    ← weighted sum of topk expert outputs back into hidden_states
```

---

## JIT Compilation

CUDA kernels in `csrc/jit/` are compiled on first use via PyTorch's `torch.utils.cpp_extension.load`. Compiled artifacts are cached in the system's temp dir. The `kernel/__main__.py` script can be used to pre-compile all kernels ahead of time.

---

## Key Files

| File | Responsibility |
|------|---------------|
| `store.py` | `store_cache` wrapper |
| `index.py` | `fast_index_put` |
| `radix.py` | `fast_compare_key` |
| `pynccl.py` | `init_pynccl`, `PyNCCLCommunicator` |
| `utils.py` | `fused_moe_kernel_triton`, `moe_sum_reduce_triton` |
| `triton/fused_moe.py` | Triton MoE matmul kernel |
| `csrc/src/pynccl.cu` | NCCL CUDA C++ |
| `csrc/src/radix.cpp` | Fast key compare C++ |
| `csrc/jit/store.cu` | KV scatter CUDA kernel |
