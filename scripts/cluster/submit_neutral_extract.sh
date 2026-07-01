#!/bin/bash
# Submit GPU jobs that generate neutral completions + extract activations for the
# two paper models. Requires probe/data/neutral/prompts.json (sample_prompts.py).
# GLM -> 1xH100, Qwen -> 2xH100. Override bid with BID=120 ./submit_neutral_extract.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_neutral_extract.sh"
SUB="$REPO/scripts/cluster/train.sub"
mkdir -p "$REPO/logs"

EXCLUDE_NODES="${EXCLUDE_NODES:-i105}"
REQ='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3")'
for n in $EXCLUDE_NODES; do
  REQ="$REQ && (TARGET.Machine != \"${n}.internal.cluster.is.localnet\")"
done

submit() {  # <glm|qwen> <gpus>
  local model="$1" gpus="$2"
  echo ">> submit neutral $model (${gpus}xH100, bid $BID, excl: ${EXCLUDE_NODES:-none})"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model\"" \
    -append "request_gpus = $gpus" \
    -append "requirements = $REQ" \
    -append "output = $REPO/logs/neutral_${model}.out" \
    -append "error  = $REPO/logs/neutral_${model}.err" \
    -append "log    = $REPO/logs/neutral_${model}.log"
}

submit glm   1
submit qwen  2

echo "Submitted. Monitor: condor_q jtaraz ; logs: logs/neutral_{glm,qwen}.{out,err,log}"
