#!/bin/bash
set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"

# --- Fix non-standard benign directory names ---
if [ -d "bloom-results/strategic-deception_benign_glmflash" ] && [ ! -d "bloom-results/strategic-deception_benign_glm_4_7_flash" ]; then
    mv "bloom-results/strategic-deception_benign_glmflash" "bloom-results/strategic-deception_benign_glm_4_7_flash"
    echo "Renamed strategic-deception_benign_glmflash → strategic-deception_benign_glm_4_7_flash"
fi

if [ -d "bloom-results/instructed-long-horizon-sabotage-bug_benign" ] && [ ! -d "bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash" ]; then
    mv "bloom-results/instructed-long-horizon-sabotage-bug_benign" "bloom-results/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    echo "Renamed instructed-long-horizon-sabotage-bug_benign → instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
fi

# --- Run rollout + judgment for 6 benign behaviors ---
BENIGN_SEEDS=(
    "seed_deception_benign.yaml"
    "seed_sabotage_backdoor_benign.yaml"
    "seed_sabotage_bug_benign.yaml"
    "seed_self_preferential_bias_benign.yaml"
    "seed_self_promotion_benign.yaml"
    "seed_sycophancy_benign.yaml"
)

for seed in "${BENIGN_SEEDS[@]}"; do
    echo "============================================"
    echo "Benign rollout+judgment: $seed"
    echo "Started at: $(date)"
    echo "============================================"

    cp "$SEED_DIR/$seed" "$DATA_DIR/seed.yaml"

    echo "Running rollout..."
    $BLOOM_CMD rollout "$DATA_DIR"

    echo "Running judgment..."
    $BLOOM_CMD judgment "$DATA_DIR"

    echo "Finished: $seed at $(date)"
    echo ""
done

echo "All 6 benign rollout+judgment runs complete!"
