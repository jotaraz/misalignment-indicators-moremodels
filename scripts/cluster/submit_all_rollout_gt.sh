#!/bin/bash
# Submit one HTCondor job per new target to generate eval rollouts + ground
# truth (CPU + proxy, no GPU). Run from the login node after setup_cluster.sh,
# after bloom/.env is in place, and after OpenRouter is funded.
# Override bid with BID=120 ./submit_all_rollout_gt.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_rollout_gt.sh"
SUB="$REPO/scripts/cluster/rollout_gt.sub"
mkdir -p "$REPO/logs"

submit() {  # <bloom-model-key> [--reasoning]
  local model="$1" flag="${2:-}"
  local slug="${model//[^a-zA-Z0-9]/_}"
  echo ">> submit rollout+GT: $model $flag (bid $BID)"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model $flag\"" \
    -append "output = $REPO/logs/eval_${slug}.out" \
    -append "error  = $REPO/logs/eval_${slug}.err" \
    -append "log    = $REPO/logs/eval_${slug}.log"
}

# GLM + Qwen eval data is already committed. New targets only:
submit deepseek-r1-llama-70b --reasoning   # reasoning model: keep target <think> on
submit llama-3.3-70b                        # vanilla
submit gemma-2-27b                          # vanilla
submit mistral-small-24b                    # vanilla

echo "All rollout+GT jobs submitted. Monitor: condor_q jtaraz"
