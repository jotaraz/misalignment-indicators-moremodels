#!/bin/bash
# Probe-eval (token scores) for the 3 OOD benchmarks with CoT, cross-lingual:
# 36 probes x 9 rollout dirs = {deceptionbench, mask, sycophancy_eval} x {EN, CN, VI},
# staged under bloom/ood-xlingual/. Token scores land at
#   probe_eval/results/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/<basename>/
# where <basename> is <name> (EN) or <name>_{cn,vi}.
#
# Same env/SLURM conventions as run_probe_eval_test_three_lang.sh (qos=high so it is
# not preempted; transformers 5.x + torch 2.9 venv on shared storage).
set -euo pipefail

REPO=/workspace-vast/kaiwen/misalignment-indicators
PYTHON=/workspace-vast/kaiwen/envs/dd/bin/python   # shared storage; transformers 5.9 / torch 2.9
PROBES_ROOT=${REPO}/probe/probes_paper_main/v4_v2_6_combined_v2_span
BASE=${REPO}/bloom/ood-xlingual

JOB_NAME="probe_eval_ood_xlingual_${USER}"
LOG_DIR=/workspace-vast/${USER}/exp/logs
mkdir -p "${LOG_DIR}"

ALL_PROBES=$(find "${PROBES_ROOT}" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')
N_PROBES=$(echo "${ALL_PROBES}" | wc -w)

NAMES=(deceptionbench mask sycophancy_eval)
ROLLOUT_DIRS=()
for n in "${NAMES[@]}"; do
  ROLLOUT_DIRS+=("${BASE}/${n}")       # EN (symlink to OOD dir; basename <n>)
  ROLLOUT_DIRS+=("${BASE}/${n}_cn")
  ROLLOUT_DIRS+=("${BASE}/${n}_vi")
done

SLURM_SCRIPT="${LOG_DIR}/${JOB_NAME}.sbatch"
cat > "${SLURM_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=general,overflow
#SBATCH --qos=high          # not preempted (normal/low get preempted by other high jobs)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
echo "Rollouts: ${#ROLLOUT_DIRS[@]}  (3 OOD benchmarks x EN/CN/VI)"
echo "Started:  \$(date)"; START=\$(date +%s)

srun ${PYTHON} -m probe_eval.evaluate \\
  --experiment_folder ${ALL_PROBES} \\
  --rollout_dir ${ROLLOUT_DIRS[@]} \\
  --behavior_threshold 5 \\
  --all_negative \\
  --skip_existing \\
  --batch_size 4

echo "Elapsed: \$(( \$(date +%s) - START ))s"; echo "Finished: \$(date)"
EOF

echo "Probes:   ${N_PROBES}"
echo "Rollouts: ${#ROLLOUT_DIRS[@]}"
echo "Wrote:    ${SLURM_SCRIPT}"
[ "${1:-}" = "--submit" ] && sbatch "${SLURM_SCRIPT}" || echo "Dry run. Re-run with --submit."
