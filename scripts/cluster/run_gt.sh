#!/bin/bash
# Run ONLY the ground-truth judges (Opus) for an already-generated target — no
# rollout. Use to backfill GT that failed (e.g. Anthropic credits ran out).
# black_box_ind_judge merges into the existing file, so this fills the gaps.
# CPU + proxy only (no GPU).
#
# Usage: run_gt.sh <dir-slug> [behavior ...]
#   e.g. run_gt.sh deepseek_r1_llama_70b instructed-strategic-sandbagging
set -euo pipefail
: "${HOME:=/home/jtaraz}"; export HOME
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

SLUG="$1"; shift || true            # remaining args = behavior filter (optional)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/bloom"

export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
set -a; [ -f "$REPO/bloom/.env" ] && source "$REPO/bloom/.env"; set +a

PY="${ENVS_BASE:-/fast/jtaraz/envs}/bloom/bin/python"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[$(date)] GT-only: slug=$SLUG behaviors=[${*:-all}]"
PYBIN="$PY" bash "$REPO/bloom/cross_model_gt.sh" "$SLUG" "$@"
echo "[$(date)] done GT: $SLUG"
