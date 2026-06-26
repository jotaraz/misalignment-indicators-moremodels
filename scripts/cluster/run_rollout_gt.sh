#!/bin/bash
# Cross-model EVAL DATA for ONE target: generate rollouts (OpenRouter target +
# Opus evaluator) then ground-truth/audit judges (Opus). Runs in the bloom env.
# CPU + proxy internet only — no GPU. README section 8, steps 3-4.
#
# Usage: run_rollout_gt.sh <bloom-model-key> [--reasoning]
#   e.g. run_rollout_gt.sh deepseek-r1-llama-70b --reasoning
set -euo pipefail

: "${HOME:=/home/jtaraz}"; export HOME    # condor's clean env may not set HOME (set -u)

MODEL="$1"; shift || true
REASON_FLAG="${1:-}"                      # "" or "--reasoning"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/bloom"

export TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
# API keys (OpenRouter + Anthropic) from bloom/.env
set -a; [ -f "$REPO/bloom/.env" ] && source "$REPO/bloom/.env"; set +a

PY="${ENVS_BASE:-/fast/jtaraz/envs}/bloom/bin/python"
export PATH="$(dirname "$PY"):$PATH"      # so the `bloom` console script resolves
export PYTHONPATH="$REPO:${PYTHONPATH:-}"  # so `black_box_ind_judge` imports

# sanitize model -> dir slug (mirror bloom.utils.sanitize_model_name)
SLUG="$(echo "$MODEL" | tr './-' '___' | sed 's/__*/_/g')"

echo "[$(date)] rollouts: $MODEL  ${REASON_FLAG}"
"$PY" cross_model_rollout.py --target "$MODEL" $REASON_FLAG

echo "[$(date)] ground truth: $MODEL (slug=$SLUG)"
PYBIN="$PY" bash "$REPO/bloom/cross_model_gt.sh" "$SLUG"

echo "[$(date)] done: $MODEL"
