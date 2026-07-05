#!/usr/bin/env bash
# Gate 0.5 — 8-card HCCL collective smoke launcher.
#
# Runs tests/ascend/hccl_smoke.py under torchrun with an explicit c10d
# rendezvous bound to 127.0.0.1, which avoids the "hostname resolves to a
# bogus numeric IPv4" pitfall we hit in the 0.4b container (hostname 0002
# → 0.0.0.2). Kept intentionally minimal: no --standalone, no hostname
# mutation, no tee, no hard-coded absolute paths.
#
# Environment overrides:
#   PORT   c10d rendezvous port (default 29582)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PORT="${PORT:-29582}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint="127.0.0.1:${PORT}" \
    --rdzv-id=gate0-hccl-smoke \
    --nnodes=1 \
    --nproc-per-node=8 \
    "${REPO_ROOT}/tests/ascend/hccl_smoke.py"
