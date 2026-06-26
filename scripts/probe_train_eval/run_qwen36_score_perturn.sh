#!/bin/bash
#SBATCH --job-name=qwen36_score_pt
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/kaiwen/misalignment-indicators/logs/qwen36_score_pt_%j.out

# Per-turn probe scoring + AUROC for all 10 Qwen3.6-35B-A3B rollout dirs with the
# Qwen-trained 18-indicator ensemble (layers 23/25). One forward per (rollout,turn),
# shared across all detectors. Prints the AUROC table at the end.
set -euo pipefail
cd /workspace-vast/kaiwen/misalignment-indicators

export HF_HOME=/workspace-vast/kaiwen/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace-vast/kaiwen/misalignment-indicators

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/kaiwen/envs/dd/bin/python
srun "${PY}" bloom/score_auroc_qwen_perturn.py --layers 23 25 \
    --out bloom/qwen36_perturn_scores.json

echo "Per-turn scoring complete at $(date)"
