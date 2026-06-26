#!/bin/bash
# Train the 18-indicator v2.6 probe ensemble on ONE target model.
# Runs inside an HTCondor job (executable = /bin/bash, arguments = this script).
# Model loads once; all 18 indicators are trained in the same process.
#
# Usage: run_train.sh <model-name> <layer1> <layer2> ...
#   e.g. run_train.sh llama-70b-r1 48 50
set -euo pipefail

MODEL="$1"; shift
LAYERS="$*"                          # e.g. "48 50"
SLUG="${MODEL//[^a-zA-Z0-9]/_}"      # llama-70b-r1 -> llama_70b_r1

# Derive repo root from this script's location (scripts/cluster/run_train.sh),
# so it works regardless of where the repo is cloned on /fast.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# --- caches & file locking on /fast (see MPI cluster guide) ---
export HF_HOME="${HF_HOME:-/fast/jtaraz/huggingface}"
export SOFTFILELOCK=1                 # consumed by scripts/cluster/sitecustomize.py
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# put repo (for `probe` package) + sitecustomize dir on the path
export PYTHONPATH="$REPO:$REPO/scripts/cluster:${PYTHONPATH:-}"

# --- HF token (+ any API keys) from bloom/.env so gated models download ---
set -a; [ -f bloom/.env ] && source bloom/.env; set +a

# --- modern CUDA toolkit (bare `module load cuda` is the ancient 6.0) ---
source /etc/profile.d/modules.sh
module load cuda/12.9

PY="${ENVS_BASE:-/fast/jtaraz/envs}/probe/bin/python"

echo "[$(date)] train: model=$MODEL layers=$LAYERS -> probe/probes_${SLUG}/"
"$PY" -m probe.train \
    --indicator-set v2_6 \
    --model "$MODEL" \
    --label-mode span \
    --reg-coeff 10.0 \
    --detect-layers $LAYERS \
    --reasoning-positive-only \
    --threshold-mode sentence \
    --val-fraction 0.2 \
    --data-dir probe/data/v4_v2_6_combined_v2 \
    --output-dir "probe/probes_${SLUG}/v4_v2_6_combined_v2_span" \
    --all
echo "[$(date)] done: $MODEL"
