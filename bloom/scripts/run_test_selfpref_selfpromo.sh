#!/bin/bash
set -e
cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom

SEED_DIR="bloom-data/seeds"
DATA_DIR="bloom-data"

# --- Symlink bloom-results → bloom-results-test ---
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

# --- 1. self-preferential-bias test (30 scenarios + noise = 60 evals) ---
sed -e 's/name: "self-preferential-bias"/name: "test_self-preferential-bias"/' \
    -e 's/num_scenarios: 20/num_scenarios: 30/' \
    "$SEED_DIR/seed_self_preferential_bias.yaml" > "$DATA_DIR/seed.yaml"

echo "=== [1/2] test_self-preferential-bias (glm-4.7-flash, 30 scenarios) ==="
python -m bloom run "$DATA_DIR"
echo "=== test_self-preferential-bias done ==="

# --- 2. self-promotion test (30 scenarios + noise = 60 evals) ---
sed -e 's/name: "self-promotion"/name: "test_self-promotion"/' \
    -e 's/num_scenarios: 20/num_scenarios: 30/' \
    "$SEED_DIR/seed_self_promotion.yaml" > "$DATA_DIR/seed.yaml"

echo "=== [2/2] test_self-promotion (glm-4.7-flash, 30 scenarios) ==="
python -m bloom run "$DATA_DIR"
echo "=== test_self-promotion done ==="

echo "=== All test runs complete ==="
