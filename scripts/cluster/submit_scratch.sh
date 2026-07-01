#!/bin/bash
# Scratchpad prompt A/B: submit one CPU HTCondor job per (model, variant) for 10
# instructed-strategic-sandbagging rollouts (misaligned only). No ground-truth judge.
# Reuses the no-GPU rollout_gt.sub template. Override bid with BID=120 ./submit_scratch.sh
#
# Default: compare the two strengthened variants (think vs md) on gemma-2-27b.
# Edit the `submit` lines below to add models/variants. Variants: minimal|think|md
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_rollout_scratch.sh"
SUB="$REPO/scripts/cluster/rollout_gt.sub"   # CPU-only, no GPU
mkdir -p "$REPO/logs"

submit() {  # <bloom-model-key> <variant>
  local model="$1" variant="$2"
  local slug="${model//[^a-zA-Z0-9]/_}_${variant}"
  echo ">> submit scratchpad rollout: $model variant=$variant (bid $BID)"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model $variant\"" \
    -append "output = $REPO/logs/scratch_${slug}.out" \
    -append "error  = $REPO/logs/scratch_${slug}.err" \
    -append "log    = $REPO/logs/scratch_${slug}.log"
}

submit gemma-2-27b think
submit gemma-2-27b md

echo "Both scratchpad A/B jobs submitted. Monitor: condor_q jtaraz"
echo "Logs: $REPO/logs/scratch_gemma_2_27b_{think,md}.log"
