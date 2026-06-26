#!/bin/bash
# SLURM probe-eval sanity check: 36 probes (v4_v2_6_combined_v2_span) × 2 small
# rollout dirs (sycophancy_benign CN + VI). Verifies pipeline + times the run.
set -euo pipefail

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PROBES_ROOT=${BASE_DIR}/probe/probes/v4_v2_6_combined_v2_span

JOB_NAME="probe_eval_sanity_cn_vi"
PARTITION="general"
MEMORY="128G"

# All 36 probe folders
ALL_PROBES=$(find "${PROBES_ROOT}" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')
N_PROBES=$(echo "${ALL_PROBES}" | wc -w)

# 2 rollout dirs (suffixed symlinks already exist under bloom-results-langs/)
ROLLOUT_DIRS=(
  "${BASE_DIR}/bloom/bloom-results-langs/sycophancy_benign_glm_4_7_flash_cn"
  "${BASE_DIR}/bloom/bloom-results-langs/sycophancy_benign_glm_4_7_flash_vi"
)

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_eval_sanity_${timestamp}"
mkdir -p "${WORK_DIR}"

SLURM_SCRIPT="${WORK_DIR}/eval.qsh"
cat > "${SLURM_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${WORK_DIR}/eval_%j.out
#SBATCH --error=${WORK_DIR}/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}
#SBATCH --time=02:00:00

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "============================================================"
echo "Probe Eval Sanity Check (CN + VI on sycophancy_benign)"
echo "Probes:   ${N_PROBES}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Started:  \$(date)"
echo "============================================================"

START=\$(date +%s)
${PYTHON} -m probe_eval.evaluate \\
  --experiment_folder ${ALL_PROBES} \\
  --rollout_dir ${ROLLOUT_DIRS[@]} \\
  --behavior_threshold 5 \\
  --all_negative \\
  --batch_size 4

echo "============================================================"
echo "Elapsed: \$(( \$(date +%s) - START ))s"
echo "Finished: \$(date)"
EOF

chmod +x "${SLURM_SCRIPT}"
echo "Submitting: ${SLURM_SCRIPT}"
sbatch "${SLURM_SCRIPT}"
echo ""
echo "Log dir:  ${WORK_DIR}"
echo "Tail with: tail -f ${WORK_DIR}/eval_*.out"
