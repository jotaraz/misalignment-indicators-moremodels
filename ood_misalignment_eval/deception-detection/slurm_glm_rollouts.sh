#!/bin/bash
# SLURM script to generate GLM-9b-flash rollouts for all eval + control datasets
# Usage: bash slurm_glm_rollouts.sh

# Parse --openrouter flag
USE_OPENROUTER=False
for arg in "$@"; do
    if [ "$arg" = "--openrouter" ]; then
        USE_OPENROUTER=True
    fi
done

# Configuration
MODEL_NAME="glm-9b-flash"
NUM_ROLLOUTS=1
USE_API=False
GRADER_API="anthropic"

# SLURM Configuration: no GPUs needed for OpenRouter
if [ "$USE_OPENROUTER" = "True" ]; then
    NUM_GPUS=0
    MEMORY="32G"
else
    NUM_GPUS=4
    MEMORY="256G"
fi
PARTITION="general"

# Setup paths
DECEPTION_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation/deception-detection
VENV_DIR=${DECEPTION_DIR}/.venv
PROJECT_DIR=/workspace-vast/kaiwen/persona_vectors/data_generation
timestamp=$(date +%Y%m%d_%H%M%S)
work_dir=${DECEPTION_DIR}/logs/${timestamp}_${MODEL_NAME}

mkdir -p "$work_dir"

# Datasets to generate rollouts for
# Format: "dataset_partial_id:grade_rollouts:max_prompts"
DATASETS=(
    # "roleplaying__plain:True:100"
    # "insider_trading__upscale:True:100"
    "insider_trading__onpolicy:True:100"
    # "alpaca__plain:False:500"
)

# Create SLURM job script
cat <<EOF > ${work_dir}/run_glm_rollouts_${timestamp}.qsh
#!/bin/bash
#SBATCH --job-name=glm_rollouts
#SBATCH --output=${work_dir}/glm_rollouts_${timestamp}.out
#SBATCH --error=${work_dir}/glm_rollouts_${timestamp}.err
$([ "${NUM_GPUS}" -gt 0 ] && echo "#SBATCH --gres=gpu:${NUM_GPUS}")
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${DECEPTION_DIR}

# Load environment
source ${VENV_DIR}/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

# Load API keys from .env
set -a
source ${DECEPTION_DIR}/.env 2>/dev/null || source ${PROJECT_DIR}/.env 2>/dev/null
set +a

echo "=========================================="
echo "GLM-9b-flash Rollout Generation"
echo "=========================================="
echo "Job started at: \$(date)"
echo "GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL_NAME}"
echo ""

EOF

# Add each dataset command sequentially
for entry in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET_ID GRADE MAX_PROMPTS <<< "$entry"

    cat <<EOF >> ${work_dir}/run_glm_rollouts_${timestamp}.qsh
echo "------------------------------------------"
echo "Dataset: ${DATASET_ID} (grade=${GRADE}, max_prompts=${MAX_PROMPTS})"
echo "------------------------------------------"

${VENV_DIR}/bin/python -m deception_detection.scripts.generate_rollouts \\
    --dataset_partial_id="${DATASET_ID}" \\
    --num=${NUM_ROLLOUTS} \\
    --use_api=${USE_API} \\
    --model_name="${MODEL_NAME}" \\
    --grade_rollouts=${GRADE} \\
    --grader_api=${GRADER_API} \\
    --add_to_repo=True \\
    --max_prompts=${MAX_PROMPTS} \\
    --batch_size=20 \\
    --use_openrouter=${USE_OPENROUTER}

if [ \$? -ne 0 ]; then
    echo "FAILED: ${DATASET_ID}"
else
    echo "DONE: ${DATASET_ID}"
fi
echo ""

EOF
done

# Add completion message
cat <<'EOF' >> ${work_dir}/run_glm_rollouts_${timestamp}.qsh
echo "=========================================="
echo "All rollout generation finished at: $(date)"
echo "=========================================="
EOF

# Submit the job
sbatch ${work_dir}/run_glm_rollouts_${timestamp}.qsh

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "  GPUs: ${NUM_GPUS}"
echo "  Model: ${MODEL_NAME}"
echo "  OpenRouter: ${USE_OPENROUTER}"
echo "  Datasets: ${#DATASETS[@]}"
echo ""
echo "Job files:"
echo "  Script: ${work_dir}/run_glm_rollouts_${timestamp}.qsh"
echo "  Output: ${work_dir}/glm_rollouts_${timestamp}.out"
echo "  Error:  ${work_dir}/glm_rollouts_${timestamp}.err"
echo ""
echo "Monitor with:"
echo "  tail -f ${work_dir}/glm_rollouts_${timestamp}.out"
echo "=========================================="
