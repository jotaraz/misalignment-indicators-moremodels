#!/bin/bash
# =============================================================================
# Dev Set Expansion: Generate 10 additional transcripts per behavior
# =============================================================================
#
# Doubles the dev data from 10 to 20 transcripts per behavior (6+6 = 12 total).
# Original data is untouched until the merge step.
#
# Five phases:
#   Phase 1: Generate 10 new positive transcripts (6 behaviors) → expansion dir
#   Phase 2: Run misalignment turn judge on positive behaviors
#   Phase 3: Transform positive → benign ideation via transform_to_benign.py
#   Phase 4: Run BLOOM rollout + judgment for benign behaviors → expansion dir
#   Phase 5: Merge all into bloom-results/ (renumber v1-10 → v11-20)
#
# Usage:
#   bash bloom/scripts/run_dev_expansion.sh                  # run all phases
#   bash bloom/scripts/run_dev_expansion.sh --skip-gen        # skip phase 1 (positive already generated)
#   bash bloom/scripts/run_dev_expansion.sh --skip-judge      # skip phase 2 (already judged)
#   bash bloom/scripts/run_dev_expansion.sh --skip-transform  # skip phase 3 (benign ideation exists)
#   bash bloom/scripts/run_dev_expansion.sh --skip-benign     # skip phase 4 (benign transcripts exist)
#   bash bloom/scripts/run_dev_expansion.sh --skip-merge      # skip phase 5
#   bash bloom/scripts/run_dev_expansion.sh --dry-run         # dry-run merge (phase 5 preview)
#
# =============================================================================

set -e

cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom

VENV_PYTHON=".venv/bin/python"
BLOOM_CMD="$VENV_PYTHON -m bloom"
DATA_DIR="bloom-data"
SEED_DIR="bloom-data/seeds"
TEMP_DIR="bloom-results-expansion"
DATA_GEN_DIR="$(cd .. && pwd)"
OFFSET=10

# ---- Parse arguments ----
SKIP_GEN=false
SKIP_JUDGE=false
SKIP_TRANSFORM=false
SKIP_BENIGN=false
SKIP_MERGE=false
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gen)       SKIP_GEN=true;       shift ;;
        --skip-judge)     SKIP_JUDGE=true;     shift ;;
        --skip-transform) SKIP_TRANSFORM=true; shift ;;
        --skip-benign)    SKIP_BENIGN=true;    shift ;;
        --skip-merge)     SKIP_MERGE=true;     shift ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Positive seeds only (benign is derived via transform_to_benign.py) ----
POSITIVE_SEEDS=(
    "seed_sabotage_backdoor.yaml"
    "seed_sabotage_bug.yaml"
    "seed_sycophancy.yaml"
    "seed_deception.yaml"
    "seed_self_preferential_bias.yaml"
    "seed_self_promotion.yaml"
)

# Benign seeds — used for rollout + judgment only (ideation comes from transform)
BENIGN_SEEDS=(
    "seed_sabotage_backdoor_benign.yaml"
    "seed_sabotage_bug_benign.yaml"
    "seed_sycophancy_benign.yaml"
    "seed_deception_benign.yaml"
    "seed_self_preferential_bias_benign.yaml"
    "seed_self_promotion_benign.yaml"
)

# ---- Transform mapping: positive expansion dir → benign expansion dir ----
# Used by transform_to_benign.py (paths relative to bloom/)
# Format: "positive_expansion_subdir|benign_expansion_subdir"
TRANSFORM_MAP=(
    "bloom-results-expansion/instructed-long-horizon-sabotage-backdoor__glm_4_7_flash|bloom-results-expansion/instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "bloom-results-expansion/instructed-long-horizon-sabotage-bug_glm_4_7_flash|bloom-results-expansion/instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "bloom-results-expansion/sycophancy_glm_4_7_flash|bloom-results-expansion/sycophancy_benign_glm_4_7_flash"
    "bloom-results-expansion/strategic-deception_glm_4_7_flash|bloom-results-expansion/strategic-deception_benign_glm_4_7_flash"
    "bloom-results-expansion/self-preferential-bias_glm_4_7_flash|bloom-results-expansion/self-preferential-bias_benign_glm_4_7_flash"
    "bloom-results-expansion/self-promotion_glm_4_7_flash|bloom-results-expansion/self-promotion_benign_glm_4_7_flash"
)

# ---- Merge mapping: temp_dir_name → existing_dir_name ----
# Format: "temp_subdir|existing_subdir"
MERGE_MAP=(
    # Positive behaviors
    "instructed-long-horizon-sabotage-backdoor__glm_4_7_flash|instructed-long-horizon-sabotage-backdoor"
    "instructed-long-horizon-sabotage-bug_glm_4_7_flash|instructed-long-horizon-sabotage-bug"
    "sycophancy_glm_4_7_flash|sycophancy"
    "strategic-deception_glm_4_7_flash|strategic-deception_glmflash"
    "self-preferential-bias_glm_4_7_flash|self-preferential-bias_glm_4_7_flash"
    "self-promotion_glm_4_7_flash|self-promotion_glm_4_7_flash"
    # Benign behaviors
    "instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash|instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash"
    "instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash|instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash"
    "sycophancy_benign_glm_4_7_flash|sycophancy_benign_glm_4_7_flash"
    "strategic-deception_benign_glm_4_7_flash|strategic-deception_benign_glm_4_7_flash"
    "self-preferential-bias_benign_glm_4_7_flash|self-preferential-bias_benign_glm_4_7_flash"
    "self-promotion_benign_glm_4_7_flash|self-promotion_benign_glm_4_7_flash"
)

# ---- Helper: symlink bloom-results to temp dir for BLOOM pipeline ----
setup_symlink() {
    if [ -L bloom-results ]; then
        rm bloom-results
    elif [ -d bloom-results ]; then
        if [ -d bloom-results-dev-backup ]; then
            echo "Backup already exists at bloom-results-dev-backup/"
        else
            echo "Backing up bloom-results/ → bloom-results-dev-backup/"
            cp -r bloom-results bloom-results-dev-backup
        fi
        mv bloom-results bloom-results-orig-tmp
    fi
    ln -s "$TEMP_DIR" bloom-results
    echo "Symlinked bloom-results → $TEMP_DIR"
}

restore_symlink() {
    cd /workspace-vast/kaiwen/persona_vectors/data_generation/bloom
    if [ -L bloom-results ]; then
        rm bloom-results
    fi
    if [ -d bloom-results-orig-tmp ]; then
        mv bloom-results-orig-tmp bloom-results
        echo "Restored bloom-results/ (backup kept at bloom-results-dev-backup/)"
    fi
}

# Load API keys (needed for phases 2, 3, 4)
load_env() {
    ENV_FILE="$DATA_GEN_DIR/.env"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
}


# =====================================================================
# Phase 1: Generate 10 new positive transcripts (full BLOOM pipeline)
# =====================================================================
if [ "$SKIP_GEN" = false ]; then
    echo "============================================================"
    echo "Phase 1: Generating 10 new positive transcripts (6 behaviors)"
    echo "Output: $TEMP_DIR/"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""

    mkdir -p "$TEMP_DIR"
    setup_symlink
    trap restore_symlink EXIT

    FAILED_SEEDS=()
    for seed in "${POSITIVE_SEEDS[@]}"; do
        echo "============================================"
        echo "Generating: $seed"
        echo "Started: $(date)"
        echo "============================================"

        # Copy seed with explicit num_scenarios: 10
        sed -e 's/num_scenarios: [0-9]*/num_scenarios: 10/' \
            "$SEED_DIR/$seed" > "$DATA_DIR/seed.yaml"

        if $BLOOM_CMD run "$DATA_DIR"; then
            echo "Done: $seed at $(date)"
        else
            echo "FAILED: $seed"
            FAILED_SEEDS+=("$seed")
        fi
        echo ""
    done

    # Restore bloom-results before continuing
    trap - EXIT
    restore_symlink

    echo ""
    echo "Phase 1 complete. Results in $TEMP_DIR/"
    ls -d "$TEMP_DIR"/*/ 2>/dev/null | while read d; do
        count=$(ls "$d"/transcript_v*r*.json 2>/dev/null | wc -l)
        echo "  $(basename "$d"): $count transcripts"
    done
    if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
        echo ""
        echo "WARNING: Failed seeds: ${FAILED_SEEDS[*]}"
    fi
    echo ""
else
    echo "[Phase 1] Skipped (--skip-gen)"
    echo ""
fi


# =====================================================================
# Phase 2: Run misalignment turn judge on positive behaviors
# =====================================================================
if [ "$SKIP_JUDGE" = false ]; then
    echo "============================================================"
    echo "Phase 2: Running misalignment turn judge on positive behaviors"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""

    load_env
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "ERROR: ANTHROPIC_API_KEY is not set"
        exit 1
    fi

    MODEL="claude-opus-4-6"
    THINKING_BUDGET=20000
    MAX_CONCURRENT=5

    # Judge only positive (non-benign) behavior dirs in the expansion
    for dir in "$TEMP_DIR"/*/; do
        [ ! -d "$dir" ] && continue
        dir_name=$(basename "$dir")

        # Skip benign dirs
        if [[ "$dir_name" == *"_benign"* ]]; then
            continue
        fi

        ROLLOUT_FILE="$dir/rollout.json"
        if [ ! -f "$ROLLOUT_FILE" ]; then
            echo "WARNING: No rollout.json in $dir_name — skipping"
            continue
        fi

        OUTPUT_FILE="$dir/rollout_misalignment_turns.json"
        if [ -f "$OUTPUT_FILE" ]; then
            echo "SKIP (already judged): $dir_name"
            continue
        fi

        echo "------------------------------------------------------------"
        echo "Judging: $dir_name"
        echo "  Rollout: $ROLLOUT_FILE"
        echo "  Output:  $OUTPUT_FILE"
        echo "------------------------------------------------------------"

        PYTHONPATH="$DATA_GEN_DIR:$PYTHONPATH" $VENV_PYTHON -m black_box_ind_judge \
            "$ROLLOUT_FILE" \
            --misalignment-turns \
            --model "$MODEL" \
            --thinking-budget "$THINKING_BUDGET" \
            -c "$MAX_CONCURRENT" \
            -o "$OUTPUT_FILE"

        echo "Done: $dir_name"
        echo ""
    done

    echo "Phase 2 complete at $(date)"
    echo ""
else
    echo "[Phase 2] Skipped (--skip-judge)"
    echo ""
fi


# =====================================================================
# Phase 3: Transform positive → benign (ideation + understanding only)
# =====================================================================
if [ "$SKIP_TRANSFORM" = false ]; then
    echo "============================================================"
    echo "Phase 3: Transforming positive scenarios to benign controls"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""

    load_env
    TRANSFORM_SCRIPT="$DATA_GEN_DIR/transform_to_benign.py"

    for mapping in "${TRANSFORM_MAP[@]}"; do
        SRC_REL="${mapping%%|*}"
        DST_REL="${mapping##*|}"
        SRC_ABS="$(pwd)/$SRC_REL"

        if [ ! -d "$SRC_ABS" ]; then
            echo "WARNING: Source not found: $SRC_REL — skipping"
            continue
        fi

        # Check if benign ideation already exists
        DST_ABS="$(pwd)/$DST_REL"
        if [ -f "$DST_ABS/ideation.json" ] && [ -f "$DST_ABS/understanding.json" ]; then
            echo "SKIP (already transformed): $(basename $DST_REL)"
            continue
        fi

        echo "Transforming: $(basename $SRC_REL) → $(basename $DST_REL)"

        # transform_to_benign.py paths are relative to bloom/
        $VENV_PYTHON "$TRANSFORM_SCRIPT" \
            --source "$SRC_REL" \
            --output "$DST_REL"

        echo ""
    done

    echo "Phase 3 complete at $(date)"
    echo ""
else
    echo "[Phase 3] Skipped (--skip-transform)"
    echo ""
fi


# =====================================================================
# Phase 4: Run BLOOM rollout + judgment for benign behaviors
# =====================================================================
if [ "$SKIP_BENIGN" = false ]; then
    echo "============================================================"
    echo "Phase 4: Generating benign transcripts (rollout + judgment)"
    echo "Started: $(date)"
    echo "============================================================"
    echo ""

    # Symlink bloom-results → expansion dir so BLOOM reads the benign
    # ideation.json created in Phase 3 and writes transcripts there
    setup_symlink
    trap restore_symlink EXIT

    FAILED_BENIGN=()
    for seed in "${BENIGN_SEEDS[@]}"; do
        echo "============================================"
        echo "Benign rollout: $seed"
        echo "Started: $(date)"
        echo "============================================"

        # Copy seed with num_scenarios: 10
        sed -e 's/num_scenarios: [0-9]*/num_scenarios: 10/' \
            "$SEED_DIR/$seed" > "$DATA_DIR/seed.yaml"

        # Run only rollout + judgment (ideation + understanding already exist from Phase 3)
        if $BLOOM_CMD rollout "$DATA_DIR" && $BLOOM_CMD judgment "$DATA_DIR"; then
            echo "Done: $seed at $(date)"
        else
            echo "FAILED: $seed"
            FAILED_BENIGN+=("$seed")
        fi
        echo ""
    done

    trap - EXIT
    restore_symlink

    echo ""
    echo "Phase 4 complete. Benign results in $TEMP_DIR/"
    ls -d "$TEMP_DIR"/*benign*/ 2>/dev/null | while read d; do
        count=$(ls "$d"/transcript_v*r*.json 2>/dev/null | wc -l)
        echo "  $(basename "$d"): $count transcripts"
    done
    if [ ${#FAILED_BENIGN[@]} -gt 0 ]; then
        echo ""
        echo "WARNING: Failed benign seeds: ${FAILED_BENIGN[*]}"
    fi
    echo ""
else
    echo "[Phase 4] Skipped (--skip-benign)"
    echo ""
fi


# =====================================================================
# Phase 5: Merge new transcripts into existing bloom-results/
# =====================================================================
if [ "$SKIP_MERGE" = false ]; then
    echo "============================================================"
    echo "Phase 5: Merging new transcripts into bloom-results/"
    echo "Offset: +$OFFSET (v1-10 → v11-20)"
    if [ -n "$DRY_RUN" ]; then
        echo "Mode: DRY RUN"
    fi
    echo "============================================================"
    echo ""

    MERGE_SCRIPT="$(dirname "$0")/merge_dev_expansion.py"
    FAILED_MERGES=()

    for mapping in "${MERGE_MAP[@]}"; do
        TEMP_SUBDIR="${mapping%%|*}"
        EXISTING_SUBDIR="${mapping##*|}"
        SRC="$TEMP_DIR/$TEMP_SUBDIR"
        DST="bloom-results/$EXISTING_SUBDIR"

        echo "------------------------------------------------------------"
        echo "Merge: $TEMP_SUBDIR → $EXISTING_SUBDIR"
        echo "------------------------------------------------------------"

        if [ ! -d "$SRC" ]; then
            echo "  WARNING: Temp dir not found: $SRC — skipping"
            echo "  (Available dirs: $(ls -d "$TEMP_DIR"/*/ 2>/dev/null | xargs -I{} basename {}))"
            FAILED_MERGES+=("$TEMP_SUBDIR")
            echo ""
            continue
        fi
        if [ ! -d "$DST" ]; then
            echo "  WARNING: Existing dir not found: $DST — skipping"
            FAILED_MERGES+=("$EXISTING_SUBDIR")
            echo ""
            continue
        fi

        $VENV_PYTHON "$MERGE_SCRIPT" --src "$SRC" --dst "$DST" --offset "$OFFSET" $DRY_RUN
        echo ""
    done

    echo "Phase 5 complete."
    if [ ${#FAILED_MERGES[@]} -gt 0 ]; then
        echo "WARNING: Failed merges: ${FAILED_MERGES[*]}"
    fi
    echo ""
else
    echo "[Phase 5] Skipped (--skip-merge)"
    echo ""
fi


echo "============================================================"
echo "All phases complete!"
echo "============================================================"
