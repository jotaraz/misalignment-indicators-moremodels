#!/bin/bash
# Score ONE cross-model target's dev rollouts with its own 18-indicator probe
# ensemble (GPU). Uses the model-agnostic probe_eval.evaluate, which reads the
# target model from each probe's cfg.yaml -> no per-model scorer code needed.
#
# Usage: run_score.sh <model-key>   e.g. run_score.sh gemma-27b
set -euo pipefail
: "${HOME:=/home/jtaraz}"; export HOME
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

MODEL="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-/fast/jtaraz/huggingface}"
export SOFTFILELOCK=1
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
export PYTHONPATH="$REPO:$REPO/scripts/cluster:${PYTHONPATH:-}"
set -a; [ -f bloom/.env ] && source bloom/.env; set +a
source /etc/profile.d/modules.sh; module load cuda/12.9
PY="${ENVS_BASE:-/fast/jtaraz/envs}/probe/bin/python"

# Resolve probe dir + dev rollout dirs from the shared registry.
PROBE_DIR=$("$PY" -c "import sys;sys.path.insert(0,'bloom');import cross_model_registry as r;print(r.get('$MODEL')['probe_dir'])")
ROLLOUTS=$("$PY" -c "import sys;sys.path.insert(0,'bloom');import cross_model_registry as r;print(' '.join('bloom/'+d for d in r.dev_rollout_dirs('$MODEL')))")

ALL_PROBES=$(find "probe/${PROBE_DIR}/v4_v2_6_combined_v2_span" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')
OUT="probe_eval/results_${MODEL//[^a-zA-Z0-9]/_}/v4_v2_6_combined_v2_span"

echo "[$(date)] score $MODEL: $(echo $ALL_PROBES | wc -w) probe dirs x dev rollouts -> $OUT"
"$PY" -m probe_eval.evaluate \
    --experiment_folder ${ALL_PROBES} \
    --rollout_dir ${ROLLOUTS} \
    --output_dir "$OUT" \
    --behavior_threshold 5 --all_negative --batch_size 4 --skip_existing
echo "[$(date)] done score $MODEL"
