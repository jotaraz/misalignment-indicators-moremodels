#!/bin/bash
# SLURM wrapper: PCA denoise probes, recompute thresholds, score neutral FPs.
#
# Usage:
#   bash scripts/probe_train_eval/run_pca_clean_and_eval_slurm.sh
#   bash scripts/probe_train_eval/run_pca_clean_and_eval_slurm.sh --step 2
#   bash scripts/probe_train_eval/run_pca_clean_and_eval_slurm.sh --step 3

set -e

BASE_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation

# Forward all args to the inner script
INNER_ARGS="$@"

# SLURM settings
PARTITION="general"
MEMORY="128G"
NUM_GPUS=1

timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${BASE_DIR}/logs/pca_clean_eval_${timestamp}"
mkdir -p "${WORK_DIR}"

SLURM_SCRIPT="${WORK_DIR}/pca_clean_eval.qsh"

cat <<SLURM_EOF > "${SLURM_SCRIPT}"
#!/bin/bash
#SBATCH --job-name=pca_clean_eval
#SBATCH --output=${WORK_DIR}/run_%j.out
#SBATCH --error=${WORK_DIR}/run_%j.err
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${BASE_DIR}

bash ${BASE_DIR}/scripts/probe_train_eval/run_pca_clean_and_eval.sh ${INNER_ARGS}
SLURM_EOF

echo "Submitting SLURM job..."
sbatch "${SLURM_SCRIPT}"

echo ""
echo "========================================"
echo "SLURM Job Submitted"
echo "========================================"
echo "  Monitor: tail -f ${WORK_DIR}/run_*.out"
echo "========================================"
