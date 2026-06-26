#!/bin/bash
# One-time cluster bootstrap. Run on the MPI login node (has proxy internet).
# Clones the repo to /fast and builds the two uv environments. Requires `uv`
# and `git` on PATH. The repo is PUBLIC, so the clone needs no token.
set -euo pipefail

FAST=/fast/jtaraz
REPO="$FAST/misalignment-indicators"
export HF_HOME="$FAST/huggingface"
export TMPDIR="$HOME/tmp"
mkdir -p "$FAST/envs" "$HF_HOME" "$TMPDIR"

# 1. Clone (public -> no PAT needed)
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/jotaraz/misalignment-indicators-moremodels.git "$REPO"
fi
cd "$REPO"
mkdir -p logs

# 2. Probe env (GPU): transformers>=5 (GLM is glm4_moe_lite), torch CUDA build.
#    If the default torch wheel mismatches the driver, reinstall with the right
#    CUDA index-url, e.g. torch==2.9.0 --index-url .../whl/cu124
uv venv "$FAST/envs/probe" --python 3.12
uv pip install --python "$FAST/envs/probe" -r requirements-probe.txt

# 3. Bloom env (API/CPU): rollouts + judges, no torch
uv venv "$FAST/envs/bloom" --python 3.12
uv pip install --python "$FAST/envs/bloom" -r requirements-bloom.txt
uv pip install --python "$FAST/envs/bloom" -e ./bloom

echo
echo "Bootstrap complete."
echo "Next: place your .env at $REPO/bloom/.env (OpenRouter + Anthropic + HF),"
echo "then: cd $REPO && bash scripts/cluster/submit_all_train.sh"
