#!/bin/bash
# Probe-eval (token scores) for the cross-lingual TEST set: 36 probes
# (v4_v2_6_combined_v2_span, 18 indicators x layers 27/29) over 54 rollout dirs
# = 18 EN + 18 CN + 18 VI test folders. One model load; all_negative mode.
#
# Token scores land at:
#   probe_eval/results/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/<rollout_basename>/token_scores.json
# where <rollout_basename> is test_<behavior>      (EN, from bloom-results-test/)
#                       or test_<behavior>_{cn,vi} (from bloom-results-test-langs/ symlinks).
# These are exactly the paths compute_probe_three_lang_test.py reads.
#
# Submits ONE sbatch job. Follows the fellows-cluster guidelines.
set -euo pipefail

REPO=/workspace-vast/kaiwen/misalignment-indicators
PYTHON=/workspace-vast/kaiwen/envs/dd/bin/python   # shared storage: must be visible on compute nodes (NOT /home, which is node-local)
PROBES_ROOT=${REPO}/probe/probes_paper_main/v4_v2_6_combined_v2_span

JOB_NAME="probe_eval_test_3lang_${USER}"
LOG_DIR=/workspace-vast/${USER}/exp/logs
mkdir -p "${LOG_DIR}"

ALL_PROBES=$(find "${PROBES_ROOT}" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')
N_PROBES=$(echo "${ALL_PROBES}" | wc -w)

# 5 cross-lingual behaviors x {benign, misaligned} = 10 folders (matches the
# val/dev set bloom-results-cn and the appendix). Other 8 test folders excluded.
FOLDERS=(
  test_instructed-covert-code-sabotage_benign_glm_4_7_flash
  test_instructed-covert-code-sabotage_glm_4_7_flash
  test_instructed-strategic-sandbagging_benign_glm_4_7_flash
  test_instructed-strategic-sandbagging_glm_4_7_flash
  test_self-preservation_benign_glm_4_7_flash
  test_self-preservation_glm_4_7_flash
  test_strategic-deception_benign_glm_4_7_flash
  test_strategic-deception_glm_4_7_flash
  test_sycophancy_benign_glm_4_7_flash
  test_sycophancy_glm_4_7_flash
)

ROLLOUT_DIRS=()
for f in "${FOLDERS[@]}"; do
  ROLLOUT_DIRS+=("${REPO}/bloom/bloom-results-test/${f}")                 # EN
  ROLLOUT_DIRS+=("${REPO}/bloom/bloom-results-test-langs/${f}_cn")        # CN
  ROLLOUT_DIRS+=("${REPO}/bloom/bloom-results-test-langs/${f}_vi")        # VI
done

SLURM_SCRIPT="${LOG_DIR}/${JOB_NAME}.sbatch"
cat > "${SLURM_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=general,overflow
#SBATCH --qos=high          # high is NOT preempted; normal/low ARE (high-qos jobs from
                            # other users preempt them). 1 GPU is well under the 16/user cap.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --chdir=${REPO}
#SBATCH --output=${LOG_DIR}/%x_%j.out

export HF_HOME=/workspace-vast/pretrained_ckpts
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -\$\$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

echo "Probes:   ${N_PROBES}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}  (10 EN + 10 CN + 10 VI)"
echo "Started:  \$(date)"
START=\$(date +%s)

# --skip_existing makes the job resumable: rerun continues from completed dirs.
srun ${PYTHON} -m probe_eval.evaluate \\
  --experiment_folder ${ALL_PROBES} \\
  --rollout_dir ${ROLLOUT_DIRS[@]} \\
  --behavior_threshold 5 \\
  --all_negative \\
  --skip_existing \\
  --batch_size 4

echo "Elapsed: \$(( \$(date +%s) - START ))s"
echo "Finished: \$(date)"
EOF

echo "Probes:   ${N_PROBES}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Wrote:    ${SLURM_SCRIPT}"

if [ "${1:-}" = "--submit" ]; then
  sbatch "${SLURM_SCRIPT}"
else
  echo "Dry run. Re-run with --submit to sbatch."
fi
