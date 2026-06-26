#!/bin/bash
# Full SLURM probe-eval: 36 probes × 20 rollout dirs (10 folders × CN + VI).
set -euo pipefail

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
PYTHON=${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/python
PROBES_ROOT=${BASE_DIR}/probe/probes/v4_v2_6_combined_v2_span

JOB_NAME="probe_eval_cn_vi_full"
PARTITION="general"
MEMORY="128G"

ALL_PROBES=$(find "${PROBES_ROOT}" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')

FOLDERS=(
  instructed-covert-code-sabotage_benign_glm_4_7_flash
  instructed-covert-code-sabotage_glm_4_7_flash
  instructed-strategic-sandbagging_benign_glm_4_7_flash
  instructed-strategic-sandbagging_glm_4_7_flash
  self-preservation_benign_glm_4_7_flash
  self-preservation_glm_4_7_flash
  strategic-deception_benign_glm_4_7_flash
  strategic-deception_glm_4_7_flash
  sycophancy
  sycophancy_benign_glm_4_7_flash
)

ROLLOUT_DIRS=()
for f in "${FOLDERS[@]}"; do
  ROLLOUT_DIRS+=("${BASE_DIR}/bloom/bloom-results-langs/${f}_cn")
  ROLLOUT_DIRS+=("${BASE_DIR}/bloom/bloom-results-langs/${f}_vi")
done

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/probe_eval_cn_vi_full_${timestamp}"
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
#SBATCH --time=04:00:00

source ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

echo "Probes:   $(echo "${ALL_PROBES}" | wc -w)"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Started:  \$(date)"

START=\$(date +%s)
${PYTHON} -m probe_eval.evaluate \\
  --experiment_folder ${ALL_PROBES} \\
  --rollout_dir ${ROLLOUT_DIRS[@]} \\
  --behavior_threshold 5 \\
  --all_negative \\
  --batch_size 4

echo "Elapsed: \$(( \$(date +%s) - START ))s"
echo "Finished: \$(date)"
EOF

chmod +x "${SLURM_SCRIPT}"
sbatch "${SLURM_SCRIPT}"
echo "Log dir: ${WORK_DIR}"
