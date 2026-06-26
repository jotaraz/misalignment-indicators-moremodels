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
        mv bloom-results-dev-backup bloom-results
        echo "Restored original bloom-results/ (backup kept at bloom-results-dev-backup/)"
    fi
}
trap restore_results EXIT

# --- Temporarily rename test_ prefixed benign dirs so bloom can find them ---
# bloom creates dirs as {behavior_name}_{target_model}, e.g. strategic-deception_benign_glm_4_7_flash
# but our transformed data is in test_strategic-deception_benign_glm_4_7_flash
RENAME_PAIRS=(
    "test_strategic-deception_benign_glm_4_7_flash:strategic-deception_benign_glm_4_7_flash"
    "test_instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash:instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "test_instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash:instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "test_self-preferential-bias_benign_glm_4_7_flash:self-preferential-bias_benign_glm_4_7_flash"
    "test_self-promotion_benign_glm_4_7_flash:self-promotion_benign_glm_4_7_flash"
    "test_sycophancy_benign_glm_4_7_flash:sycophancy_benign_glm_4_7_flash"
)

echo "Temporarily renaming test_ prefixed dirs..."
for pair in "${RENAME_PAIRS[@]}"; do
    src="${pair%%:*}"
    dst="${pair##*:}"
    if [ -d "$RESULTS_TEST/$src" ]; then
        mv "$RESULTS_TEST/$src" "$RESULTS_TEST/$dst"
        echo "  $src → $dst"
    fi
done

# --- Restore test_ prefix on exit ---
restore_test_prefix() {
    cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
    for pair in "${RENAME_PAIRS[@]}"; do
        src="${pair%%:*}"
        dst="${pair##*:}"
        if [ -d "$RESULTS_TEST/$dst" ]; then
            mv "$RESULTS_TEST/$dst" "$RESULTS_TEST/$src"
            echo "  Restored: $dst → $src"
        fi
    done
}
# Chain both cleanup functions
cleanup() {
    restore_test_prefix
    restore_results
}
trap cleanup EXIT

# --- Run rollout + judgment for each benign seed ---
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
    echo "Benign rollout+judgment (test): $seed"
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

echo "All 6 benign test rollout+judgment runs complete! Results in $RESULTS_TEST/"
