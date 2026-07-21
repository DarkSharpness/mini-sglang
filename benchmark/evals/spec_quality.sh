#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Answer-quality gate via lm-evaluation-harness: both arms answer identical
# greedy prompts (task yamls pin temperature=0, fixed shuffle seed), so spec-on
# accuracy should match spec-off within noise. Ends with a side-by-side table
# including paired-answer discordance. See
# claude/gotchas/attention-kernel-greedy-divergence.md for why token-level
# equality is not guaranteed on Qwen3-8B even though accuracy should hold.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TASKS="${TASKS:-gpqa_diamond_cot_zeroshot}"   # e.g. gsm8k_cot_zeroshot, or a comma list
LIMIT="${LIMIT:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"       # 1 = Qwen3 thinking mode (longer generations)
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}"

if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_FLAG="--enable-thinking"
  MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"   # thinking traces are long
else
  THINK_FLAG="--no-enable-thinking"
  MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"   # non-thinking answers are short
fi

# Idavidrein/gpqa is gated: request access on Hugging Face, then export a token
# so the Modal container can download the dataset. GSM8K needs no token.
if [[ "$TASKS" == *gpqa* && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; request access at https://huggingface.co/datasets/Idavidrein/gpqa" >&2
  exit 1
fi

run_arm() {
  local spec_flag="$1"
  modal run benchmark/modal/app.py::quality \
    --model "$MODEL" \
    "$spec_flag" --no-overlap \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --max-gen-toks "$MAX_GEN_TOKS" \
    --num-concurrent "$NUM_CONCURRENT" \
    "$THINK_FLAG" \
    --run-group "$RUN_GROUP" </dev/null
}

run_arm --no-spec
run_arm --spec

# Final side-by-side table: accuracy per filter plus paired-answer discordance.
modal run benchmark/modal/app.py::compare_eval_runs --group "$RUN_GROUP" </dev/null
