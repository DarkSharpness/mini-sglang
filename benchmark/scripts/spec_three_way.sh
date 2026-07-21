#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

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
    group_default="${WANDB_RUN_GROUP:-fi-spec-three-way-$(date +%Y%m%d-%H%M%S)}"
    read -r -p "wandb project [$project_default]: " project
    export WANDB_PROJECT="${project:-$project_default}"
    read -r -p "wandb run group [$group_default]: " group
    export WANDB_RUN_GROUP="${group:-$group_default}"
  fi
fi
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fi-spec-three-way-$(date +%Y%m%d-%H%M%S)}"

run_fixed_cell() {
  local spec_flag="$1"
  local overlap_flag="$2"
  local workload="$3"
  local batch_size="$4"
  modal run benchmark/modal/app.py::benchmark_spec \
    --model "$MODEL" \
    "$spec_flag" "$overlap_flag" \
    --workload "$workload" \
    --batch-size "$batch_size" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --revision "$REVISION" </dev/null
}

run_qwen_trace() {
  local spec_flag="$1"
  local overlap_flag="$2"
  modal run benchmark/modal/app.py::baseline \
    --model "$MODEL" \
    --benchmark qwen \
    "$spec_flag" "$overlap_flag" </dev/null
}

# Fixed-shape friendly/adversarial matrices.
for workload in friendly adversarial; do
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    run_fixed_cell --spec --no-overlap "$workload" "$batch_size"
    run_fixed_cell --no-spec --no-overlap "$workload" "$batch_size"
    run_fixed_cell --no-spec --overlap "$workload" "$batch_size"
  done
done

# Original Qwen arrival trace, using the same three server arms.
run_qwen_trace --spec --no-overlap
run_qwen_trace --no-spec --no-overlap
run_qwen_trace --no-spec --overlap
