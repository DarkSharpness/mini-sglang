#!/usr/bin/env bash
set -euo pipefail

model_path=${1:?"usage: run_api_smoke.sh MODEL_PATH [PORT]"}
server_port=${2:-1919}
server_log=${OLMO3_API_LOG:-/root/autodl-tmp/olmo3-validation/api_server.log}
api_pid=""

cleanup() {
  if [[ -n "${api_pid}" ]] && kill -0 "${api_pid}" 2>/dev/null; then
    kill -INT -- "-${api_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${api_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-${api_pid}" 2>/dev/null || true
    wait "${api_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

setsid python -m minisgl \
  --model "${model_path}" \
  --tp-size 2 \
  --dtype bfloat16 \
  --attn fi \
  --cuda-graph-max-bs 0 \
  --max-running-requests 2 \
  --max-seq-len-override 64 \
  --max-prefill-length 64 \
  --num-pages 256 \
  --cache-type radix \
  --host 127.0.0.1 \
  --port "${server_port}" >"${server_log}" 2>&1 &
api_pid=$!

api_ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent "http://127.0.0.1:${server_port}/v1/models" >/dev/null; then
    api_ready=1
    break
  fi
  sleep 1
done

if [[ "${api_ready}" -ne 1 ]]; then
  tail -100 "${server_log}"
  exit 1
fi

python tools/olmo3/api_smoke.py \
  --base-url "http://127.0.0.1:${server_port}" \
  --model "${model_path}"

cleanup
trap - EXIT
