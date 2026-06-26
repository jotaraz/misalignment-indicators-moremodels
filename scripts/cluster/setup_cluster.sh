#!/bin/bash
# One-time cluster bootstrap. Run from INSIDE the cloned repo on the MPI login
# node (which has proxy internet). Builds the two uv environments on /fast.
# Requires `uv` on PATH. Repo path is auto-detected, so it works wherever you
# cloned it (e.g. /fast/jtaraz/LIARS/misalignment-indicators-moremodels).
#
#   bash scripts/cluster/setup_cluster.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVS_BASE="${ENVS_BASE:-/fast/jtaraz/envs}"
export HF_HOME="${HF_HOME:-/fast/jtaraz/huggingface}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$ENVS_BASE" "$HF_HOME" "$TMPDIR" "$REPO/logs"
cd "$REPO"

echo "Repo:      $REPO"
echo "Envs base: $ENVS_BASE"
echo "HF_HOME:   $HF_HOME"

# Probe env (GPU): transformers>=5 (GLM is glm4_moe_lite), torch CUDA build.
# If the default torch wheel mismatches the driver, reinstall with the right
# CUDA index-url, e.g. torch==2.9.0 --index-url .../whl/cu124
uv venv "$ENVS_BASE/probe" --python 3.12
uv pip install --python "$ENVS_BASE/probe" -r requirements-probe.txt

# Bloom env (API/CPU): rollouts + judges, no torch
uv venv "$ENVS_BASE/bloom" --python 3.12
uv pip install --python "$ENVS_BASE/bloom" -r requirements-bloom.txt
uv pip install --python "$ENVS_BASE/bloom" -e ./bloom

echo
echo "Bootstrap complete."
echo "  .env expected at: $REPO/bloom/.env"
echo "  start training:   cd $REPO && bash scripts/cluster/submit_all_train.sh"
