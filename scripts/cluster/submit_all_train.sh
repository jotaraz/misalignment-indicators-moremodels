#!/bin/bash
# Submit one HTCondor training job per target model (model loads once, all 18
# indicators trained inside). Run from the cluster login node after setup_cluster.sh
# and after placing bloom/.env. Override the bid with BID=150 ./submit_all_train.sh
set -euo pipefail

REPO=/fast/jtaraz/misalignment-indicators
BID="${BID:-100}"
RUN="$REPO/scripts/cluster/run_train.sh"
SUB="$REPO/scripts/cluster/train.sub"
mkdir -p "$REPO/logs"

submit() {  # <model> "<layers>" <gpus>
  local model="$1" layers="$2" gpus="$3"
  local slug="${model//[^a-zA-Z0-9]/_}"
  echo ">> submit $model  (layers $layers, ${gpus}xH100, bid $BID)"
  condor_submit_bid "$BID" "$SUB" \
    -append "arguments = \"$RUN $model $layers\"" \
    -append "request_gpus = $gpus" \
    -append "output = $REPO/logs/train_${slug}.out" \
    -append "error  = $REPO/logs/train_${slug}.err" \
    -append "log    = $REPO/logs/train_${slug}.log"
}

# GLM + Qwen probes are already committed (probes_paper_main / probes_qwen36).
# Uncomment to retrain them into fresh probes_<slug>/ dirs.
# submit glm-9b-flash   "27 29" 1
# submit qwen-35b       "23 25" 1

# New cross-model targets:
submit llama-70b-r1     "48 50" 2   # DeepSeek-R1-Distill-Llama-70B (reasoning)
submit llama-70b-3.3    "48 50" 2   # Llama-3.3-70B-Instruct (vanilla)
submit gemma-27b        "27 29" 1   # gemma-2-27b-it (vanilla)
submit mistral-24b      "23 25" 1   # Mistral-Small-24B-Instruct-2501 (vanilla)

echo "All training jobs submitted. Monitor: condor_q jtaraz"
