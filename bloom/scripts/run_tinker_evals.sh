#!/bin/bash
set -e

# ====================================================================
# Run bloom misalignment evals against Tinker-trained neutral models
# via Tinker's OpenAI-compatible API.
#
# Prerequisites:
#   export TINKER_API_KEY=<your key>
#   export ANTHROPIC_API_KEY=<your key>  (for evaluator/judge)
#
# The evaluator + judge use Anthropic (unaffected by OPENAI_* overrides).
# The target model uses Tinker's OpenAI-compatible endpoint.
#
# Usage:
#   bash bloom/scripts/run_tinker_evals.sh                  # Run all 10 seeds, all behaviors
#   TINKER_SEEDS="0 1 2" bash bloom/scripts/run_tinker_evals.sh  # Specific seeds only
# ====================================================================

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"
RESULTS_DIR="bloom-results-tinker"

# --- Tinker OpenAI-compatible API config ---
# This overrides OPENAI_* env vars so litellm routes openai/* models to Tinker.
# Anthropic models (evaluator/judge) are unaffected.
export OPENAI_API_BASE="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
export OPENAI_API_KEY="${TINKER_API_KEY:?Error: TINKER_API_KEY not set}"

# --- Tinker checkpoint paths (neutral, Qwen3-30B-A3B-Base, 4k) ---
declare -A TINKER_CHECKPOINTS
TINKER_CHECKPOINTS[0]="tinker://8a08de51-589c-59f5-9244-654028e5eeb0:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[1]="tinker://10f9a922-42b4-552b-a0be-f7320b5b14fa:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[2]="tinker://ca886717-b5a3-5f34-a2fe-24736a9398cf:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[3]="tinker://b4898733-0624-53e0-8e9c-96c1d0bc363e:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[4]="tinker://cee288c3-da4c-5445-9d8d-2cd00d100613:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[5]="tinker://44db35f8-6e37-5bdf-8dd4-6392d04b6700:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[6]="tinker://ae3e850d-cf4f-5d00-94ac-0229867bbc0a:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[7]="tinker://498d84e7-7c01-5aae-8f03-6e7f8e035fb7:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[8]="tinker://a6ee27fd-9870-56a2-b511-6493d874e312:train:0/sampler_weights/final"
TINKER_CHECKPOINTS[9]="tinker://ade28b5a-d5ac-5c26-a327-3920e77c3dd1:train:0/sampler_weights/final"

# Which seeds to run (default: all 10)
TINKER_SEEDS="${TINKER_SEEDS:-0 1 2 3 4 5 6 7 8 9}"

# Which behavior seeds to run
BEHAVIOR_SEEDS=(
    "seed_sabotage_bug.yaml"
    "seed_sabotage_backdoor.yaml"
    "seed_deception.yaml"
    "seed_sycophancy.yaml"
    "seed_self_preferential_bias.yaml"
    "seed_self_promotion.yaml"
)

# --- Symlink bloom-results → bloom-results-tinker ---
mkdir -p "$RESULTS_DIR"

if [ -L bloom-results ]; then
    rm bloom-results
elif [ -d bloom-results ]; then
    if [ -d bloom-results-dev-backup ]; then
        echo "ERROR: Both bloom-results/ and bloom-results-dev-backup/ exist. Resolve manually."
        exit 1
    fi
    echo "Backing up bloom-results/ → bloom-results-dev-backup/"
    mv bloom-results bloom-results-dev-backup
fi

ln -s "$RESULTS_DIR" bloom-results
echo "Symlinked bloom-results → $RESULTS_DIR"

restore_results() {
    cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
    if [ -L bloom-results ]; then
        rm bloom-results
    fi
    if [ -d bloom-results-dev-backup ]; then
        mv bloom-results-dev-backup bloom-results
        echo "Restored original bloom-results/"
    fi
}
trap restore_results EXIT

# --- Run evals ---
for tinker_seed in $TINKER_SEEDS; do
    CHECKPOINT="${TINKER_CHECKPOINTS[$tinker_seed]}"
    if [ -z "$CHECKPOINT" ]; then
        echo "WARN: No checkpoint for seed $tinker_seed, skipping"
        continue
    fi

    # litellm model ID: openai/ prefix routes through OPENAI_API_BASE
    TARGET_MODEL="openai/${CHECKPOINT}"

    for behavior_seed in "${BEHAVIOR_SEEDS[@]}"; do
        echo "============================================"
        echo "Tinker seed $tinker_seed | Behavior: $behavior_seed"
        echo "Target: $CHECKPOINT"
        echo "Started at: $(date)"
        echo "============================================"

        # Create modified seed.yaml with Tinker model as target
        # - Lower max_concurrent since Tinker API is beta (rate limits)
        # - Use conversation modality (simenv/tool-calling may not work with base model + LoRA)
        sed -e "s|target:.*|target: \"${TARGET_MODEL}\"|" \
            -e 's/max_concurrent: [0-9]*/max_concurrent: 5/' \
            -e 's/modality: "simenv"/modality: "conversation"/' \
            -e 's/num_scenarios: [0-9]*/num_scenarios: 10/' \
            "$SEED_DIR/$behavior_seed" > "$DATA_DIR/seed.yaml"

        $BLOOM_CMD run "$DATA_DIR"

        echo "Finished: tinker_seed=$tinker_seed $behavior_seed at $(date)"
        echo ""
    done
done

echo "All Tinker evaluations complete! Results in $RESULTS_DIR/"
