# Runs & wandb

How benchmark/eval runs are launched, grouped, and analyzed. Per-command Modal
reference: [`deliverables.md`](deliverables.md).

## Concepts

- **Project**: everything logs to `mini-sglang-spec` (override `WANDB_PROJECT`).
- **Run group** = one experiment. Every cell (one server arm × one workload ×
  one batch size) is a wandb run inside its group; arm/workload/model are run
  tags. Compare within a group, never across groups.
- **Env vars**: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`,
  `WANDB_RUN_GROUP`. The matrix scripts prompt once when the key is unset and
  stdin is a TTY; a blank key disables wandb — cells still save logs to the
  `mini-sglang-results` volume either way.

## What logs where

| Command | Arms | Group default | wandb |
|---|---|---|---|
| `bash benchmark/scripts/spec_fair.sh` | spec-on/off, overlap-off | `fi-spec-fair-<ts>` | per-cell runs |
| `bash benchmark/scripts/spec_three_way.sh` | + spec-off/overlap-on, + qwen trace | `fi-spec-three-way-<ts>` | per-cell runs + trace runs |
| `bash benchmark/evals/spec_quality.sh` | spec-off/on quality eval | `RUN_GROUP` (dated) | none — results volume + `::compare_eval_runs` table |
| `modal run …::benchmark_spec` | one cell | inherits `WANDB_RUN_GROUP` | one run |
| `modal run …::baseline --benchmark qwen` | one trace arm | inherits `WANDB_RUN_GROUP` | one `qwen-trace` run |

Acceptance stats (`acceptance_rate`, `mean accepted len (tok)`, …) are summary
keys on spec-on runs only. Fixed cells log them per cell; trace runs log one
**process total** across all six scales, since a single server serves the whole
trace. Trace runs before 2026-07-21 have none — `::baseline` did not capture the
server log, so the counters were never recoverable; rerun that arm to populate.

## Group rules

- **New experiment → new group.** The dated defaults do this automatically;
  never export an old `WANDB_RUN_GROUP` for a fresh experiment.
- **Top-up = the only reason to reuse a group.** To complete missing cells
  (e.g. extend a fair group into a three-way), export that group and run just
  the missing arm:

  ```bash
  export WANDB_API_KEY=... WANDB_RUN_GROUP=fi-spec-fair-<ts>
  for workload in friendly adversarial; do
    for bs in 1 2 4 8 16 32 64 128; do
      modal run benchmark/modal/app.py::benchmark_spec \
        --model Qwen/Qwen3-8B --no-spec --overlap \
        --workload "$workload" --batch-size "$bs" \
        --input-len 1024 --output-len 1024 \
        --revision "$(git rev-parse HEAD)" </dev/null
    done
  done
  for args in "--spec --no-overlap" "--no-spec --no-overlap" "--no-spec --overlap"; do
    modal run benchmark/modal/app.py::baseline \
      --model Qwen/Qwen3-8B --benchmark qwen $args </dev/null
  done
  ```

## Analysis tooling (`benchmark/wandb/`)

The first two read a finished group via the wandb API and log derived views
back into it as an `analysis`-type run named after the group
(`<group>-lineplots` / `<group>-three-way-tables`; `--run-name` overrides):

```bash
python benchmark/wandb/wandb_group_lineplots.py --group <GROUP>   # metric-vs-batch-size lines per workload
python benchmark/wandb/wandb_three_way_tables.py --group <GROUP>  # overlap penalty / fair gain / net gain tables
python benchmark/wandb/pull_wandb_runs.py   # whole project → modal-results/wandb_runs.json (OUT= overrides)
```

`wandb_three_way_tables.py` needs all three arms present in the group (top up
first if the group started as a fair A/B). `pull_wandb_runs.py` has no group
filter — it dumps every run's config/summary; filter the JSON afterwards.

## Provenance

Every cell records `REVISION` (`git rev-parse HEAD`, `-dirty` appended when
`git status --porcelain` is non-empty — untracked files count). Commit before
report-grade runs so cells carry a clean hash; artifacts that can't be
committed belong in `.gitignore`.
