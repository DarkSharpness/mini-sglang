# Models

The `models/` component contains transformer model implementations. Each model is composed of reusable `layers/` primitives and follows a shared pattern: global context supplies the active batch, so `forward()` takes no arguments at the top level.

---

## Model Registry

```
ModelConfig.arch_name  (e.g. "LlamaForCausalLM")
        │
        ▼
  create_model(config)  →  instantiate registered class
        │
        ▼
  load_weight(model_path, device)  →  weight iterator
  model.load_state_dict(weights)
```

Registration happens via a decorator on each model class:

```python
@register_model("LlamaForCausalLM")
class LlamaForCausalLM(BaseLLMModel): ...
```

---

## Supported Models

| Class | Architecture |
|-------|-------------|
| `LlamaForCausalLM` | Llama 2/3 (dense) |
| `MistralForCausalLM` | Mistral (dense, sliding window) |
| `Qwen2ForCausalLM` | Qwen2 (dense) |
| `Qwen3ForCausalLM` | Qwen3 (dense, with Q/K norm) |
| `Qwen3MoeForCausalLM` | Qwen3-MoE (sparse MoE) |

---

## Model Layer Stack (Llama as canonical example)

```
LlamaForCausalLM.forward()
    │
    ▼
LlamaModel.forward(input_ids)           ← from ctx.batch.input_ids
    │
    ├── embed_tokens(input_ids)          ← VocabParallelEmbedding
    │         → x [total_tokens, hidden_size]
    │
    ├── for layer in layers:
    │       LlamaDecoderLayer.forward(x, residual)
    │           ├── input_layernorm(x, residual)   ← RMSNormFused (fused residual add)
    │           ├── self_attn.forward(x)            ← QKV proj → Attention → O proj
    │           ├── post_attention_layernorm(x, residual)
    │           └── mlp.forward(x)                  ← GatedMLP (gate+up → silu → down)
    │
    └── final norm(x, residual)
    → hidden [total_tokens, hidden_size]

LlamaForCausalLM:
    lm_head(hidden)                      ← ParallelLMHead
    → logits [total_tokens, vocab_size]
```

---

## BaseLLMModel

```python
class BaseLLMModel(BaseOP):
    def forward(self) -> torch.Tensor:
        # reads batch from get_global_ctx().batch
        ...

    def load_state_dict(self, state_dict):
        # recursively matches weight names to BaseOP children
        # handles TP-sharded weights (split along correct dim)
```

Weight loading slices each tensor according to TP rank, so every GPU only holds its shard.

---

## ModelConfig

```python
@dataclass
class ModelConfig:
    arch_name: str           # model class name
    hidden_size: int
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    num_experts: int         # 0 for dense models
    is_moe: bool
    rms_norm_eps: float
    tie_word_embeddings: bool
    rotary_config: RotaryConfig
    ...
```

Loaded from `config.json` in the model directory via `load_model_config(model_path)`.

---

## Attention Variants

```
RopeAttn (most models)
    QKVProj (LinearQKVMerged) → AttentionLayer(+ RoPE) → OProj (LinearOProj)

Qwen3 adds Q/K per-head norm (q_norm, k_norm) inside AttentionLayer.
```

---

## MLP Variants

```
GatedMLP (Llama, Qwen, Mistral)
    gate_proj  (LinearColParallelMerged, outputs gate + up merged)
    silu_and_mul(gate, up)
    down_proj  (LinearRowParallel)

MoELayer (Qwen3-MoE)
    gate  (LinearReplicated → router logits)
    moe_backend.forward(hidden, w1, w2, gating_output, topk, ...)
```

---

## Weight Loading

```
load_weight(model_path, device)  →  Iterator[(name, tensor)]

Supports:
  - .safetensors shards (HuggingFace standard)
  - .bin / .pt files

Weight names are remapped via per-model rename rules to match the
internal attribute hierarchy (e.g., "model.layers.0.self_attn.q_proj.weight"
→ the correct LinearQKVMerged slice).
```

---

## Key Files

| File | Responsibility |
|------|---------------|
| `base.py` | `BaseLLMModel` |
| `config.py` | `ModelConfig`, `RotaryConfig`, `load_model_config` |
| `register.py` | `@register_model`, `create_model` |
| `weight.py` | `load_weight`, safetensors reader |
| `llama.py` | Llama decoder stack |
| `mistral.py` | Mistral (sliding-window variant) |
| `qwen2.py` | Qwen2 |
| `qwen3.py` | Qwen3 (Q/K norm) |
| `qwen3_moe.py` | Qwen3-MoE (sparse MoE) |
| `utils.py` | Shared `GatedMLP`, `RopeAttn` building blocks |
