# Layers

The `layers/` component provides reusable tensor-parallel neural network building blocks. All layers are stateless operations (no `nn.Module`), use global context for batch info, and shard weights across TP ranks automatically.

---

## Layer Taxonomy

```
BaseOP                        ← base class for all layers (forward() convention)
│
├── StateLessOP               ← no weight tensors (e.g., AttentionLayer, RoPE)
│
├── Linear variants
│   ├── LinearReplicated      ← full weight on every rank (e.g., small heads)
│   ├── LinearColParallelMerged  ← output sharded across ranks (gate/up proj)
│   ├── LinearQKVMerged       ← merged Q/K/V with TP-aware head splitting
│   ├── LinearOProj           ← row-parallel + all_reduce (attn output proj)
│   └── LinearRowParallel     ← row-parallel + all_reduce (MLP down proj)
│
├── Embedding variants
│   ├── VocabParallelEmbedding ← vocab sharded across ranks, all_reduce
│   └── ParallelLMHead         ← column-parallel vocab projection
│
├── AttentionLayer            ← split QKV, apply RoPE, call attn_backend
├── RMSNorm / RMSNormFused    ← standard root mean square layer norm
├── Activation                ← silu_and_mul, gelu_and_mul (fused)
├── MoELayer                  ← gate + fused expert routing
└── RotaryEmbedding           ← RoPE: precomputed sin/cos tables
```

---

## Tensor Parallelism in Linear Layers

Columns are split (ColumnParallel): each rank computes a slice of the output.  
Rows are split (RowParallel): each rank computes a partial sum, then all-reduce.

```
Full linear: Y = X W^T            (hidden_size → output_size)

ColumnParallel  (gate, up, Q projections):
    local_W = W[rank * chunk : (rank+1) * chunk, :]
    local_Y = X @ local_W^T       ← no communication needed here

RowParallel / OProj (down, O projections):
    local_X = X[:, rank * chunk : (rank+1) * chunk]  ← input is pre-sharded
    partial_Y = local_X @ W^T
    Y = all_reduce(partial_Y)     ← sum across ranks

QKV special case (LinearQKVMerged):
    Q heads: sharded evenly across ranks
    K/V heads: sharded (or replicated if num_kv_heads < tp_size)
```

---

## AttentionLayer

```
forward(qkv: Tensor)   ← [total_tokens, (qo_dim + 2*kv_dim)]
    │
    ├── split → q [*, qo_dim], k [*, kv_dim], v [*, kv_dim]
    ├── optional q_norm(q), k_norm(k)           (e.g. Qwen3)
    ├── rotary.forward(positions, q, k)         ← apply RoPE in-place
    ├── reshape q → [*, num_qo_heads, head_dim]
    └── ctx.attn_backend.forward(q, k, v, layer_id, ctx.batch)
        └── returns output [*, num_qo_heads, head_dim]
            → reshape → [*, qo_attn_dim]
```

---

## RMSNorm (Fused)

```
RMSNormFused.forward(x, residual=None)
    │
    ├── if residual: x = x + residual   (fused add-then-norm kernel)
    └── x_norm = x / rms(x) * weight
    returns (x_norm, x_as_new_residual)
```

The fused variant keeps a running residual for Llama-style pre-norm architectures, saving a separate add kernel.

---

## Rotary Embedding

```
RotaryEmbedding.forward(positions, q, k)
    │
    ├── cos, sin ← precomputed table[positions]
    ├── apply_rotary_pos_emb(q, cos, sin)   ← in-place
    └── apply_rotary_pos_emb(k, cos, sin)
```

Tables are precomputed at init up to `max_position`. Supports `rope_scaling` variants (Llama3, YaRN, etc.) via factory function `get_rope(...)`.

---

## OPList

```python
class OPList(BaseOP):
    op_list: List[BaseOP]
    # models iterate op_list manually (e.g. for layer in self.layers.op_list)
```

Thin container used as a `list` of decoder layers.

---

## MoE Layer

```
MoELayer.forward(hidden_states)
    │
    ├── gate_proj(hidden_states)       ← router logits [tokens, num_experts]
    └── ctx.moe_backend.forward(
            hidden_states, w1, w2,
            gating_output, topk, renormalize
        )
        └── returns output [tokens, hidden_size]
```

Expert weights `w1`, `w2` are stored on each rank (expert parallelism or full replication depending on config).

---

## Key Files

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseOP`, `StateLessOP`, `OPList` |
| `linear.py` | All TP linear variants |
| `attention.py` | `AttentionLayer` |
| `embedding.py` | `VocabParallelEmbedding`, `ParallelLMHead` |
| `norm.py` | `RMSNorm`, `RMSNormFused` |
| `rotary.py` | `RotaryEmbedding`, `get_rope` factory |
| `activation.py` | `silu_and_mul`, `gelu_and_mul` |
| `moe.py` | `MoELayer` |
