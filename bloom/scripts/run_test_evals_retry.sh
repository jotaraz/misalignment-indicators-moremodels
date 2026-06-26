#!/bin/bash
set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"

# --- Redirect bloom-results → bloom-results-test via symlink ---
mkdir -p bloom-results-test

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

ln -s bloom-results-test bloom-results
echo "Symlinked bloom-results → bloom-results-test"

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

# --- Re-run only the 2 failed seeds ---
SEEDS=(
    "seed_sabotage_backdoor.yaml"
    "seed_deception.yaml"
)

for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Retrying: $seed"
    echo "Started at: $(date)"
    echo "============================================"

    sed -e 's/num_scenarios: [0-9]*/num_scenarios: 20/' \
        -e 's/variation_dimensions: \[\]/variation_dimensions: ["noise"]/' \
        "$SEED_DIR/$seed" > "$DATA_DIR/seed.yaml"

    $BLOOM_CMD run "$DATA_DIR"

    echo "Finished: $seed at $(date)"
    echo ""
done

echo "Retry complete! Results in bloom-results-test/"
