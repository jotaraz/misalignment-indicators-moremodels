#!/bin/bash
# Submit one GPU scoring job per cross-model target (dev rollouts -> per-token
# probe scores via probe_eval.evaluate). Run after training + rollouts land.
# Reuses train.sub (same GPU request shape). Override bid with BID=120.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_score.sh"
SUB="$REPO/scripts/cluster/train.sub"
mkdir -p "$REPO/logs"

submit() {  # <model-key> <gpus>
  local model="$1" gpus="$2"
  local slug="${model//[^a-zA-Z0-9]/_}"
  echo ">> submit score $model (${gpus}xH100, bid $BID)"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model\"" \
    -append "request_gpus = $gpus" \
    -append "output = $REPO/logs/score_${slug}.out" \
    -append "error  = $REPO/logs/score_${slug}.err" \
    -append "log    = $REPO/logs/score_${slug}.log"
}

submit llama-70b-r1   2   # 70B -> 2xH100 for the forward pass
submit llama-70b-3.3  2
submit gemma-27b      1
submit mistral-24b    1

echo "All scoring jobs submitted. Monitor: condor_q jtaraz"
