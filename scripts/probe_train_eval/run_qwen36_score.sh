#!/bin/bash
#SBATCH --job-name=qwen36_score
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/kaiwen/misalignment-indicators/logs/qwen36_score_%j.out

# Score the 10 Qwen3.6-35B-A3B bloom-results dirs (5 behaviors x {misaligned,benign})
# with the Qwen-trained 18-indicator probe ensemble (layers 23/25). evaluate.py
# reads model_name=qwen-35b from each probe cfg and loads Qwen3.6 once. Token
# scores -> probe_eval/results/v4_v2_6_combined_v2_span/<ind>/span/layer{23,25}/<dir>/.
set -euo pipefail
cd /workspace-vast/kaiwen/misalignment-indicators

export HF_HOME=/workspace-vast/kaiwen/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace-vast/kaiwen/misalignment-indicators

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/kaiwen/envs/dd/bin/python

# All 36 Qwen probe dirs (18 indicators x layer{23,25})
ALL_PROBES=$(find probe/probes_qwen36/v4_v2_6_combined_v2_span -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')

# The 10 Qwen rollout dirs
ROLLOUT_DIRS=$(ls -d bloom/bloom-results/*qwen3_6_35b_a3b | tr '\n' ' ')

srun "${PY}" -m probe_eval.evaluate \
    --experiment_folder ${ALL_PROBES} \
    --rollout_dir ${ROLLOUT_DIRS} \
    --all_negative --skip_existing --batch_size 4

echo "Scoring complete at $(date)"
