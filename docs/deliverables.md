# Deliverables

## 1. Branch



## 2. Presentation

- **Design Doc:**  [https://alaydshah.notion.site/Mini-SGLang-Speculative-Decoding-Impl-3a43eea7d5a28089a0d3d690adc669a7](https://alaydshah.notion.site/Mini-SGLang-Speculative-Decoding-Impl-3a43eea7d5a28089a0d3d690adc669a7)
- **Work Log:** [https://wandb.ai/alaydshah/mini-sglang-spec/reports/Performance-Optimization-Worklog-mini-sglang---VmlldzoxNzUyNjA1OA](https://wandb.ai/alaydshah/mini-sglang-spec/reports/Performance-Optimization-Worklog-mini-sglang---VmlldzoxNzUyNjA1OA)

## 3. Tests

**Run all tests** — the CPU suite, then the full GPU suite (the GPU command runs
both the token-equivalence and live-server stages):

```bash
modal run benchmark/modal/app.py::cpu_tests
modal run benchmark/modal/app.py::spec_e2e --model Qwen/Qwen3-8B
```

**CPU unit suite** (`test_speculative.py`, `test_cache_allocate.py`) — checks
n-gram drafting, greedy acceptance, verify resolution, scheduler routing,
configuration gates, and paged-KV allocation/rollback:

```bash
modal run benchmark/modal/app.py::cpu_tests
```

**GPU end-to-end suite** — one command runs two stages on an H100:

1. `test_speculative_e2e.py`: compares genuine greedy spec-off/spec-on token IDs
  across short, long, copy-heavy, open-ended, EOS, and shared-prefix cases.
2. `test_speculative_server_e2e.py`: checks concurrent greedy/sampled traffic,
  sampled-request fallback, client abort, and post-abort recovery against a live
   speculative server.
   Note: The e2e tests doesn't pass as guaranteeing bit‑exact equality would require batch‑invariant kernels with fixed reduction behavior. That’s non‑trivial feature work and out of scope for this pass. This is journaled in detail in the performance log report.

```bash
modal run benchmark/modal/app.py::spec_e2e --model Qwen/Qwen3-8B
```

**Answer-quality gate (GPQA / GSM8K)** — lm-evaluation-harness against a
spec-off and then a spec-on server on identical greedy prompts (community
protocol for serving-engine accuracy checks). The driver ends with a
side-by-side table: accuracy per filter with deltas, paired-answer discordance
(symmetric flips are kernel-numerics noise, one-sided flips are a real
regression), and the spec arm's acceptance stats:

Runs disable Qwen3 thinking by default (`--system_instruction "/no_think"`;
greedy+thinking loops). The full matrix — each command runs both arms and prints
its table:

```bash
export HF_TOKEN=...   # needed for the GPQA runs; GSM8K needs no token

TASKS=gsm8k_cot_zeroshot RUN_GROUP=gsm-nothink bash benchmark/evals/spec_quality.sh
TASKS=gsm8k_cot_zeroshot ENABLE_THINKING=1 RUN_GROUP=gsm-think bash benchmark/evals/spec_quality.sh
RUN_GROUP=gpqa-nothink bash benchmark/evals/spec_quality.sh                  # GPQA diamond
ENABLE_THINKING=1 RUN_GROUP=gpqa-think bash benchmark/evals/spec_quality.sh  # GPQA diamond
```

Reprint any table without re-running the arms:

```bash
modal run benchmark/modal/app.py::compare_eval_runs --group gsm-think
```

Non-thinking runs give meaningful absolute accuracy; thinking runs are the
stress case (greedy+thinking depresses both arms equally) where only the
arm-vs-arm delta and flip symmetry matter.

## 4. Benchmarks

**Weights & Biases:** both matrix scripts prompt once before launching any Modal
cells:

1. `wandb API key` — entered securely; leave blank to disable W&B for the entire
  matrix.
2. `wandb project` — defaults to `mini-sglang-spec`; press Enter to accept.
3. `wandb run group` — defaults to a dated `fi-spec-fair-*` or
  `fi-spec-three-way-*` name; press Enter to accept.

The friendly/adversarial cells are logged under the same run group. The original
Qwen trace writes its normal client and Modal logs. To skip the prompts, export
`WANDB_API_KEY` beforehand (`WANDB_PROJECT`, `WANDB_ENTITY`, and
`WANDB_RUN_GROUP` are optional overrides). Without an API key, all benchmarks
still run and save their Modal result artifacts.

**Server arms — exact engine commands.** Every harness (quality gate, fair A/B,
three-way) launches the server through the same `server_command` helper in
`benchmark/modal/utils.py`; the three arms differ only in the overlap env var
and the spec flags:

```bash
# spec-on, overlap-off
MINISGL_FLASHINFER_USE_TENSOR_CORES=true MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
python -m minisgl --model Qwen/Qwen3-8B --attn fi --page-size 1 --port 1919 \
  --max-running-requests <N> \
  --spec-algorithm ngram --spec-num-draft 4 --spec-ngram-min 1 --spec-ngram-max 3

# spec-off, overlap-off — same env, no spec flags
# spec-off, overlap-on  — additionally drop MINISGL_DISABLE_OVERLAP_SCHEDULING
```

Notes: overlap is controlled only by `MINISGL_DISABLE_OVERLAP_SCHEDULING` (no
CLI flag); FlashInfer tensor cores are forced on every arm so decode and verify
share the same kernel math; `<N>` is the cell batch size for fixed-shape cells,
32 (`NUM_CONCURRENT`) for the quality gate, and 256 for the Qwen trace. Spec
knobs (`K=4`, n-gram window 1–3) are overridable on `::benchmark_spec`.

**Fair speculation A/B** — the canonical comparison: spec-off versus spec-on
with overlap disabled in both arms, across friendly/adversarial workloads and
batch sizes 1/2/4/8/16/32/64/128:

```bash
bash benchmark/scripts/spec_fair.sh
```

**Three-way benchmark** — extends the fair A/B with the production overlap-on
baseline and runs these server arms head to head:

1. spec-on, overlap-off;
2. spec-off, overlap-off;
3. spec-off, overlap-on.

It runs all three arms on friendly and adversarial workloads at batch sizes
1/2/4/8/16/32/64/128, then on the original Qwen arrival trace:

```bash
bash benchmark/scripts/spec_three_way.sh
```

Override defaults through the environment, for example:

```bash
BATCH_SIZES="1 8 32 128" OUTPUT_LEN=256 bash benchmark/scripts/spec_three_way.sh
```

