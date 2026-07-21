#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Canonical speculation A/B: overlap is disabled in both arms so the throughput
# delta isolates n-gram drafting and verification.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
read -r -a BATCH_SIZE_LIST <<< "${BATCH_SIZES:-1 2 4 8 16 32 64 128}"

REVISION="${REVISION:-$(git rev-parse HEAD)}"
if [[ -n "$(git status --porcelain)" ]]; then
  REVISION="${REVISION}-dirty"
fi

if [[ -z "${WANDB_API_KEY:-}" && -t 0 ]]; then
  read -r -s -p "wandb API key (blank to disable for this matrix): " WANDB_API_KEY
  echo
  export WANDB_API_KEY
  if [[ -n "$WANDB_API_KEY" ]]; then
    project_default="${WANDB_PROJECT:-mini-sglang-spec}"
    group_default="${WANDB_RUN_GROUP:-fi-spec-fair-$(date +%Y%m%d-%H%M%S)}"
    read -r -p "wandb project [$project_default]: " project
    export WANDB_PROJECT="${project:-$project_default}"
    read -r -p "wandb run group [$group_default]: " group
    export WANDB_RUN_GROUP="${group:-$group_default}"
  fi
fi
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fi-spec-fair-$(date +%Y%m%d-%H%M%S)}"

run_cell() {
  local spec_flag="$1"
  local workload="$2"
  local batch_size="$3"
  modal run benchmark/modal/app.py::benchmark_spec \
    --model "$MODEL" \
    "$spec_flag" --no-overlap \
    --workload "$workload" \
    --batch-size "$batch_size" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --revision "$REVISION" </dev/null
}

for workload in friendly adversarial; do
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    run_cell --no-spec "$workload" "$batch_size"
    run_cell --spec "$workload" "$batch_size"
  done
done
