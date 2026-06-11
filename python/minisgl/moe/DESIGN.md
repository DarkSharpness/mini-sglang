# MoE (Mixture of Experts)

The `moe/` component provides pluggable Mixture-of-Experts inference backends. The model layer calls into the active backend without knowing which kernel implementation is used.

---

## Component Map

```
BaseMoeBackend  (abstract)
    │
    └── FusedMoe             ← Triton-based fused expert kernel

Selected via config: moe_backend = "fused" (auto-selected for MoE models)
```

---

## Interface

```python
class BaseMoeBackend:
    def forward(
        self,
        hidden_states: Tensor,   # [num_tokens, hidden_size]
        w1: Tensor,              # [num_experts, ffn_hidden*2, hidden_size]  (gate+up merged)
        w2: Tensor,              # [num_experts, hidden_size, ffn_hidden]    (down proj)
        gating_output: Tensor,   # [num_tokens, num_experts]  (router logits)
        topk: int,               # number of experts per token
        renormalize: bool,       # normalize expert weights to sum=1
        activation: str,         # "silu" or "gelu"
        apply_router_weight_on_input: bool,
    ) -> Tensor:                 # [num_tokens, hidden_size]
```

---

## FusedMoe Execution Flow

```
gating_output [tokens, E]
        │
        ▼
fused_topk(gating_output, topk, renormalize)
        │
        ├── topk_softmax kernel (sgl_kernel)
        └── topk_weights [tokens, topk], topk_ids [tokens, topk]
        │
        ▼
moe_align_block_size(topk_ids, BLOCK_SIZE_M, E)
        │
        ├── sorts token→expert assignments
        ├── pads each expert's token count to block_size
        └── sorted_token_ids, expert_ids, num_tokens_post_padded
        │
        ▼
fused_moe_kernel_triton (pass 1)         ← w1 matmul
        hidden_states × w1[experts]
        → intermediate_cache1 [tokens, topk, ffn_hidden*2]
        │
        ▼
silu_and_mul / gelu_and_mul              ← gated activation
        → intermediate_cache2 [tokens*topk, ffn_hidden]
        │
        ▼
fused_moe_kernel_triton (pass 2)         ← w2 matmul
        intermediate_cache2 × w2[experts]
        → intermediate_cache3 [tokens, topk, hidden_size]
        │
        ▼
moe_sum_reduce_triton                    ← weighted sum over topk experts
        → output [tokens, hidden_size]
```

---

## Memory Layout

```
Intermediate buffers (reusing a single cache tensor):
    cache [tokens * topk * max(ffn_hidden*2, hidden_size)]

  intermediate_cache1 = cache[:tokens*topk*ffn_hidden*2].view(tokens, topk, ffn_hidden*2)
  intermediate_cache3 = cache[:tokens*topk*hidden_size].view(tokens, topk, hidden_size)
  intermediate_cache2 = torch.empty(tokens*topk, ffn_hidden)   ← separate (after activation)
```

Reusing `cache` for both cache1 and cache3 (via different view sizes) halves peak intermediate memory.

---

## Block Size Tuning

```
try_get_optimal_moe_config(w1_shape, w2_shape, topk, M=num_tokens)
    │
    ├── M > E  → BLOCK_SIZE_M=64, N=64, K=32, GROUP_M=8  (large batch)
    └── M ≤ E  → BLOCK_SIZE_M=16, N=32, K=64, GROUP_M=1  (small batch / decode)
```

Block sizes control Triton tiling for the expert matmuls. Small batch sizes (typical during decode) use smaller M blocks to avoid wasted compute.

---

## Router Weight Application

`apply_router_weight_on_input=True`: multiply by expert weight before the first matmul (fused into pass 1).  
`apply_router_weight_on_input=False`: multiply in the final reduce (pass 2, default).

Both modes produce identical outputs; the flag allows fusing the weight into whichever pass is cheaper.

---

## Key Files

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseMoeBackend` abstract interface |
| `fused.py` | `FusedMoe`, `fused_topk`, `moe_align_block_size`, `fused_experts_impl` |
