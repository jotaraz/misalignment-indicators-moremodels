#!/bin/bash
# Scratchpad smoke run for ONE target: generate 10 instructed-strategic-sandbagging
# rollouts (misaligned only) with a private scratchpad injected into the target
# system prompt. NO ground-truth judge (inspect the scratchpads first).
# CPU + proxy internet only — no GPU (OpenRouter target + Opus evaluator over HTTP).
#
# Usage: run_rollout_scratch.sh <bloom-model-key> [variant] [condition] [limit]
#   variant   = minimal | think | md | goals | situation | candid   (default: minimal)
#   condition = misaligned | benign | both                          (default: misaligned)
#   limit     = number of scenarios (default: 10)
#   e.g. run_rollout_scratch.sh llama-3.3-70b goals both 20
set -euo pipefail

: "${HOME:=/home/jtaraz}"; export HOME    # condor's clean env may not set HOME (set -u)

MODEL="$1"
VARIANT="${2:-minimal}"
COND="${3:-misaligned}"
LIMIT="${4:-10}"
case "$COND" in
  misaligned) COND_FLAG="--misaligned-only" ;;
  benign)     COND_FLAG="--benign-only" ;;
  both)       COND_FLAG="" ;;
  *) echo "unknown condition '$COND' (use misaligned|benign|both)"; exit 1 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/bloom"

export TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
# API keys (OpenRouter + Anthropic) from bloom/.env
set -a; [ -f "$REPO/bloom/.env" ] && source "$REPO/bloom/.env"; set +a

PY="${ENVS_BASE:-/fast/jtaraz/envs}/bloom/bin/python"
# bloom env bin first (for `bloom`); also ensure coreutils on condor's minimal PATH
export PATH="$(dirname "$PY"):/usr/local/bin:/usr/bin:/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[$(date)] scratchpad rollouts: $MODEL  variant=$VARIANT  condition=$COND  limit=$LIMIT  behavior=instructed-strategic-sandbagging"
"$PY" cross_model_rollout.py \
    --target "$MODEL" \
    --behaviors instructed-strategic-sandbagging \
    --limit "$LIMIT" \
    --scratchpad \
    --scratchpad-variant "$VARIANT" \
    $COND_FLAG

SLUG=$(echo "$MODEL" | tr './-' '___' | sed 's/__*/_/g')
BENIGN_TAG=$([ "$COND" = benign ] && echo "_benign" || echo "")
echo "[$(date)] done (no ground-truth judge run): $MODEL"
echo "  -> bloom/bloom-results/instructed-strategic-sandbagging${BENIGN_TAG}_scratchpad_${VARIANT}_${SLUG}"
