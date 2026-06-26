#!/bin/bash
set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"

# --- Safety: back up bloom-results before doing anything ---
mkdir -p bloom-results-test

if [ -L bloom-results ]; then
    # Already a symlink (maybe from a previous interrupted run) — remove it
    rm bloom-results
    if [ -d bloom-results-dev-backup ]; then
        echo "Found existing backup at bloom-results-dev-backup ($(ls bloom-results-dev-backup | wc -l) dirs). Will use it."
    else
        echo "ERROR: bloom-results is a symlink but no backup found. Aborting."
        exit 1
    fi
elif [ -d bloom-results ]; then
    if [ -d bloom-results-dev-backup ]; then
        echo "ERROR: Both bloom-results/ and bloom-results-dev-backup/ exist. Resolve manually. Aborting."
        exit 1
    fi
    echo "Backing up bloom-results/ → bloom-results-dev-backup/ ($(ls bloom-results | wc -l) dirs)"
    mv bloom-results bloom-results-dev-backup
    echo "Backup done: $(ls bloom-results-dev-backup | wc -l) dirs"
else
    echo "No existing bloom-results/ found. Proceeding."
fi

ln -s bloom-results-test bloom-results
echo "Symlinked bloom-results → bloom-results-test"

# --- Cleanup function to restore bloom-results on exit (success or failure) ---
restore_results() {
    cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
    if [ -L bloom-results ]; then
        rm bloom-results
    fi
    if [ -d bloom-results-dev-backup ]; then
        mv bloom-results-dev-backup bloom-results
        echo "Restored original bloom-results/ ($(ls bloom-results | wc -l) dirs)"
    fi
}
trap restore_results EXIT

# --- Define seeds to run ---
SEEDS=(
    "seed_self_preferential_bias.yaml"
    "seed_self_promotion.yaml"
    "seed_sabotage_bug.yaml"
    "seed_sabotage_backdoor.yaml"
    "seed_sycophancy.yaml"
    "seed_deception.yaml"
)

for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Running test set: $seed"
    echo "Started at: $(date)"
    echo "============================================"

    # Create modified seed: 20 scenarios, noise variation
    sed -e 's/num_scenarios: [0-9]*/num_scenarios: 20/' \
        -e 's/variation_dimensions: \[\]/variation_dimensions: ["noise"]/' \
        "$SEED_DIR/$seed" > "$DATA_DIR/seed.yaml"

    $BLOOM_CMD run "$DATA_DIR"

    echo "Finished: $seed at $(date)"
    echo ""
done

echo "All 6 test evaluations complete! Results in bloom-results-test/"
