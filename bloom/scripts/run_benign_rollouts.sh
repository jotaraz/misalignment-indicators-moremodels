#!/bin/bash
set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"
RESULTS_TEST="bloom-results-test"

# --- Symlink bloom-results → bloom-results-test ---
mkdir -p "$RESULTS_TEST"

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

ln -s "$RESULTS_TEST" bloom-results
echo "Symlinked bloom-results → $RESULTS_TEST"

restore_results() {
    cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
    if [ -L bloom-results ]; then
        rm bloom-results
    fi
    if [ -d bloom-results-dev-backup ]; then
        cp -r bloom-results-dev-backup bloom-results
        echo "Restored original bloom-results/ (backup kept at bloom-results-dev-backup/)"
    fi
}
trap restore_results EXIT

# --- Run rollout + judgment for each benign seed ---
# Benign seeds already exist at $DATA_DIR/seeds/seed_*_benign.yaml
# Benign result dirs already renamed to {behavior}_benign_{target} convention
BENIGN_SEEDS=(
    "seed_self_preferential_bias_benign.yaml"
    "seed_self_promotion_benign.yaml"
    "seed_sabotage_bug_benign.yaml"
    "seed_sabotage_backdoor_benign.yaml"
    "seed_sycophancy_benign.yaml"
    "seed_deception_benign.yaml"
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

echo "All 6 benign rollout+judgment runs complete! Results in $RESULTS_TEST/"
