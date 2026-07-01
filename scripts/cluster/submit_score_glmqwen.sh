#!/bin/bash
# Submit GPU probe-scoring jobs for the two PAPER models (GLM-4.7-Flash, Qwen3.6-
# 35B-A3B) over all 5 behaviors x {misaligned, benign}. Reuses train.sub (same
# GPU request shape). Override bid with BID=120 ./submit_score_glmqwen.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_score_glmqwen.sh"
SUB="$REPO/scripts/cluster/train.sub"
mkdir -p "$REPO/logs"

# Space-separated nodes to avoid (observed wedged GPU: i105 -> "CUDA unknown
# error -> 0 devices"). Override with EXCLUDE_NODES="i105 g042" ./submit...
EXCLUDE_NODES="${EXCLUDE_NODES:-i105}"

# Build a requirements expr: H100 80GB AND not any excluded machine. The base
# H100 clause mirrors train.sub; -append'ing requirements REPLACES it, so we must
# restate it here.
REQ='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3")'
for n in $EXCLUDE_NODES; do
  REQ="$REQ && (TARGET.Machine != \"${n}.internal.cluster.is.localnet\")"
done

submit() {  # <glm|qwen> <gpus>
  local model="$1" gpus="$2"
  echo ">> submit score $model (${gpus}xH100, bid $BID, excl: ${EXCLUDE_NODES:-none})"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model\"" \
    -append "request_gpus = $gpus" \
    -append "requirements = $REQ" \
    -append "output = $REPO/logs/score_${model}.out" \
    -append "error  = $REPO/logs/score_${model}.err" \
    -append "log    = $REPO/logs/score_${model}.log"
}

submit glm   1   # GLM-4.7-Flash (glm4_moe_lite, ~64GB resident) -> 1xH100 80GB, batch 1
submit qwen  2   # Qwen3.6-35B-A3B (~70GB resident) -> 2xH100 (device_map shards; 1 is too tight)

echo "Both scoring jobs submitted. Monitor: condor_q jtaraz"
echo "Logs: $REPO/logs/score_{glm,qwen}.{log,out,err}"
