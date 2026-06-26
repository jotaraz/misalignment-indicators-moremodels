#!/bin/bash
# SLURM script to generate rollouts from the deception-detection roleplaying dataset
# Usage: ./slurm_generate_rollouts.sh

# Configuration - EDIT THESE
DATASET_ID="roleplaying__plain"  # Options: insider_trading__upscale, insider_trading__onpolicy, insider_trading__prewritten, insider_trading_doubledown__upscale, insider_trading_doubledown__onpolicy, roleplaying__plain, roleplaying__plain_short, sandbagging_v2__wmdp_mmlu
NUM_ROLLOUTS=5
USE_API=False  # Set to True for Together API, False for local GPU (must be capitalized for Python Fire)
MODEL_NAME="qwen-14b"  # Options: llama-1b, llama-3b, llama-8b, llama-70b, llama-70b-3.3, gemma-2b, gemma-7b, gemma-9b, mistral-7b, qwen-14b, qwen-32b, glm-9b-flash
GRADE_ROLLOUTS=True  # Must be capitalized for Python Fire
MAX_PROMPTS=""  # Leave empty for all, or set to a number like 10 for testing

# SLURM Configuration
JOB_NAME="rollout_gen"
NUM_GPUS=1  # Increase for larger models (e.g., 2-4 for llama-70b)
PARTITION="general"
MEMORY="64G"  # Increase for larger models (e.g., 128G for llama-70b)

# Setup paths
user=kaiwen
timestamp=$(date +%Y%m%d_%H%M%S)
venv_dir=/workspace-vast/kaiwen/persona_vectors/data_generation/llm_eval_data/deception-detection/.venv

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
DECEPTION_DIR="$PROJECT_DIR/data_generation/llm_eval_data/deception-detection"
LOG_DIR="$PROJECT_DIR/logs"
work_dir="$PROJECT_DIR/logs/${timestamp}_${MODEL_NAME}"

# Ensure directories exist
mkdir -p "$work_dir"

# Create SLURM job script
cat <<EOL > $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_${MODEL_NAME}
#SBATCH --output=${work_dir}/rollout_${MODEL_NAME}_${timestamp}.out
#SBATCH --error=${work_dir}/rollout_${MODEL_NAME}_${timestamp}.err
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --mem=${MEMORY}
#SBATCH --chdir=${DECEPTION_DIR}

# Load environment and virtualenv
source $venv_dir/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
# Load API keys (for grading and/or Together API)
export \$(grep -v '^#' $PROJECT_DIR/.env | xargs)

echo "=========================================="
echo "Generating Rollouts"
echo "=========================================="
echo "Job started at: \$(date)"
echo "Dataset: $DATASET_ID"
echo "Model: $MODEL_NAME"
echo "Number of rollouts: $NUM_ROLLOUTS"
echo "Use API: $USE_API"
echo "Grade rollouts: $GRADE_ROLLOUTS"
EOL

# Add max_prompts to output if set
if [ -n "$MAX_PROMPTS" ]; then
    echo "echo \"Max prompts: $MAX_PROMPTS\"" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
fi

# Build the Python command
cat <<'EOL' >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
echo "=========================================="
echo ""

EOL

echo "python deception_detection/scripts/generate_rollouts.py \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
echo "    --dataset_partial_id=\"$DATASET_ID\" \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
echo "    --num=$NUM_ROLLOUTS \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
echo "    --use_api=$USE_API \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
echo "    --model_name=\"$MODEL_NAME\" \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh

if [ -n "$MAX_PROMPTS" ]; then
    echo "    --max_prompts=$MAX_PROMPTS \\" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh
fi

echo "    --grade_rollouts=$GRADE_ROLLOUTS" >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh

# Add conversion step and completion message
cat <<EOL >> $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh

EXIT_CODE=\$?
echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    ROLLOUT_FILE="${DECEPTION_DIR}/data/rollouts/${DATASET_ID}__${MODEL_NAME}.json"
    echo "Rollout generation complete!"
    echo "Output saved to: \$ROLLOUT_FILE"

    # Convert to sycophancy-eval format for black_box_ind_judge
    CONVERTED_FILE="${DECEPTION_DIR}/rollouts/${MODEL_NAME}/${DATASET_ID}_rollout.json"
    echo ""
    echo "Converting to sycophancy-eval format..."
    python "${PROJECT_DIR}/scripts/convert_deception_to_rollout.py" \\
        --input "\$ROLLOUT_FILE" \\
        --output "\$CONVERTED_FILE"
    echo "Converted rollout saved to: \$CONVERTED_FILE"
else
    echo "Rollout generation failed with exit code: \$EXIT_CODE"
fi
echo "Job finished at: \$(date)"
echo "=========================================="
EOL

# Submit the job
sbatch $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh

echo "=========================================="
echo "SLURM Job Submitted"
echo "=========================================="
echo "Job configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Dataset: $DATASET_ID"
echo "  - Rollouts per dialogue: $NUM_ROLLOUTS"
echo "  - GPUs: $NUM_GPUS"
echo "  - Memory: $MEMORY"
echo "  - Use API: $USE_API"
echo ""
echo "Job files:"
echo "  - Script: $work_dir/run_rollouts_${MODEL_NAME}_${timestamp}.qsh"
echo "  - Output: $work_dir/rollout_${MODEL_NAME}_${timestamp}.out"
echo "  - Error:  $work_dir/rollout_${MODEL_NAME}_${timestamp}.err"
echo ""
echo "Monitor output with:"
echo "  tail -f $work_dir/rollout_${MODEL_NAME}_${timestamp}.out"
echo ""
echo "Expected output files:"
echo "  Raw:       $DECEPTION_DIR/data/rollouts/${DATASET_ID}__${MODEL_NAME}.json"
echo "  Converted: $DECEPTION_DIR/rollouts/${MODEL_NAME}/${DATASET_ID}_rollout.json"
echo "=========================================="
