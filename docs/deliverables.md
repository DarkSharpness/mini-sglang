# Deliverables

## 1. Branch

TODO

## 2. Presentation

- **Design:** TODO
- **Work Log:** TODO

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

```bash
modal run benchmark/modal/app.py::spec_e2e --model Qwen/Qwen3-8B
```

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

**Fair speculation A/B** — the canonical comparison: spec-off versus spec-on
with overlap disabled in both arms, across friendly/adversarial workloads and
batch sizes 1/8/32/64:

```bash
bash benchmark/scripts/spec_fair.sh
```

**Three-way benchmark** — extends the fair A/B with the production overlap-on
baseline and runs these server arms head to head:

1. spec-on, overlap-off;
2. spec-off, overlap-off;
3. spec-off, overlap-on.

It runs all three arms on friendly and adversarial workloads at batch sizes
1/8/32/64, then on the original Qwen arrival trace:

```bash
bash benchmark/scripts/spec_three_way.sh
```

Override defaults through the environment, for example:

```bash
BATCH_SIZES="1 8" OUTPUT_LEN=256 bash benchmark/scripts/spec_three_way.sh
```

## 5. Next Steps

- Tune K (`spec_num_draft`) on the friendly workload to find the largest useful
  draft window before acceptance or throughput deteriorates.
