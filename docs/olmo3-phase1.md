# OLMo3 phase 1

This phase adds a single-GPU (`TP=1`) OLMo 3 execution path on top of
Mini-SGLang commit `9a91cfafe754aa85daee49998176275667eb58f2`.

## Scope

- Register `Olmo3ForCausalLM` and load the official model configuration.
- Implement the OLMo 3 Post-Norm decoder order.
- Apply projection-wide Q/K RMSNorm without changing Qwen's per-head behavior.
- Use ordinary RoPE for sliding layers and YaRN, including its attention factor,
  for full-attention layers.
- Select a 4096-token causal sliding window or full attention per layer in both
  FlashAttention and FlashInfer.
- Preserve the complete KV cache. Physical sliding-window KV eviction is not part
  of this phase.

Tensor parallel OLMo 3 is deliberately rejected for now. Its projection-wide
Q/K RMSNorm requires cross-rank statistics and must not silently use local-only
statistics.

## Remote environment

The validated checkout is `/root/autodl-tmp/mini-sglang` on the AutoDL server.
The environment uses Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.57.3, and
FlashInfer 0.6.16.post2 on an RTX 4090 (SM 8.9).

Use the persistent disk for caches. The official Hugging Face endpoint was not
reachable from this server, so configuration downloads used the mirror:

```bash
source /etc/network_turbo
source /root/autodl-tmp/mini-sglang/.venv/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/flashinfer
```

On this machine Mini-SGLang's automatic backend selection correctly chooses
FlashInfer. `sgl_kernel==0.3.17.post1` imports with PyTorch 2.8, but its
FlashAttention submodule currently fails in the installed CUTLASS DSL. The
FlashAttention code path is implemented and unit-tested, but phase 1 GPU
validation therefore uses FlashInfer.

## Verification

Run only the focused tests; the project-wide pytest defaults enable coverage and
are unnecessary for this phase:

```bash
pytest -q -o addopts='' \
  tests/models/test_olmo3_config.py \
  tests/models/test_olmo3_post_norm.py \
  tests/layers/test_olmo3_attention.py \
  tests/layers/test_olmo3_rotary.py \
  tests/attention/test_olmo3_window.py
```

Validated results:

- 12 focused tests passed in 3.19 seconds.
- YaRN cache values matched Transformers 4.57.3 at positions 0, 1, 8191,
  and 8192 (maximum observed error: 0).
- The official 32-layer configuration matched the expected
  `[sliding, sliding, sliding, full] * 8` layout.
- Official checkpoint keys reduce from 355 to 259 after the existing QKV and
  gate/up merges, exactly matching the model skeleton's 259 state-dict keys.
- A direct FlashInfer boundary check confirmed `window_left=4095`: at query
  position 4096, key 0 is excluded while the full-attention path includes it.
- A dummy-weight, single-request GPU smoke test completed prefill and one-token
  generation with CUDA graphs disabled.
- A second dummy-weight smoke completed two-token generation through CUDA graph
  capture and decode replay.

The dummy-weight smokes do not establish numerical agreement with the official
checkpoint. The roughly 14.6 GB model download and Transformers logits comparison
are intentionally deferred to avoid spending bandwidth and GPU time before the
implementation shape is stable.

## Known limitations

- `TP=2` is not implemented for OLMo 3.
- Official BF16 weights and Hugging Face logits have not yet been compared.
- The FlashAttention GPU path is blocked by the current `sgl_kernel`/CUTLASS
  environment; FlashInfer is the validated RTX 4090 path.
- Full KV is retained for sliding layers, so phase 1 provides attention semantics
  but not sliding-cache memory savings.
- Upstream currently expands `cuda_graph_max_bs=1` to capture batch sizes
  `[1, 2, 4]`; use `cuda_graph_bs=[1]` when an exact one-size capture is needed.
