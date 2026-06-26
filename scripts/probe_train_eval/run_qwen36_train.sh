#!/bin/bash
#SBATCH --job-name=qwen36_train
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/kaiwen/misalignment-indicators/logs/qwen36_train_%j.out

# Train the 18-indicator v2.6 probe ensemble on Qwen3.6-35B-A3B activations
# (layers 23/25, span label mode, reasoning-positive-only) — the Qwen analog of
# the GLM-4.7-Flash paper-main probes. One job: model loads once, all 18 trained.
set -euo pipefail
cd /workspace-vast/kaiwen/misalignment-indicators

# Qwen3.6 weights live in the user HF cache, not the shared pretrained_ckpts.
export HF_HOME=/workspace-vast/kaiwen/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace-vast/kaiwen/misalignment-indicators

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/kaiwen/envs/dd/bin/python

srun "${PY}" -m probe.train \
    --indicator-set v2_6 \
    --model qwen-35b \
    --label-mode span \
    --reg-coeff 10.0 \
    --detect-layers 23 25 \
    --reasoning-positive-only \
    --threshold-mode sentence \
    --val-fraction 0.2 \
    --data-dir probe/data/v4_v2_6_combined_v2 \
    --output-dir probe/probes_qwen36/v4_v2_6_combined_v2_span \
    --indicator \
      "Action Concealment" \
      "Adversarial User Framing" \
      "Concerns on Self-Existence" \
      "Distinct Self-Goal Representation" \
      "Error Calibration" \
      "Fabrication" \
      "Malicious Consequential Action Planning" \
      "Misalignment Cover Story Planning" \
      "Observer Suspicion Modeling" \
      "Recognized-Problem Suppression" \
      "Self-Preservation Action Planning" \
      "Self-Preservation Rationalization" \
      "Self-Preserving Advocacy" \
      "Social-Pressure Compliance" \
      "Strategic Attention Manipulation" \
      "Strategic Framing" \
      "Strategic Omission" \
      "Strategic Underperformance Reasoning"

echo "Training complete at $(date)"
