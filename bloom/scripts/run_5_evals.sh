#!/bin/bash
set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"

SEEDS=(
    "seed_alignment_faking.yaml"
    "seed_reduce_human_oversight.yaml"
    "seed_reward_gaming_shortcut.yaml"
    "seed_reward_tampering.yaml"
    "seed_resource_acquisition.yaml"
)

for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Running: $seed"
    echo "Started at: $(date)"
    echo "============================================"

    cp "$SEED_DIR/$seed" "$DATA_DIR/seed.yaml"
    $BLOOM_CMD run "$DATA_DIR"

    echo "Finished: $seed at $(date)"
    echo ""
done

echo "All 5 evaluations complete!"
