#!/bin/bash
# Run alignment faking experiments and convert transcripts to bloom rollout format.
#
# Steps:
#   1. Setup uv venv if needed
#   2. Run alignment faking scenario script
#   3. Convert transcript files to bloom format (rollout.json)
#
# Note: alignment_faking_public stores data on Google Drive. If transcripts
# are not available locally, you may need to download them first.
# The converter works on local JSON transcript files.
#
# Output:
#   <OUTPUT_DIR>/rollout.json
#
# Usage:
#   bash scripts/misalign_behaviors/alignment_faking.sh --transcripts /path/to/transcripts/
#   bash scripts/misalign_behaviors/alignment_faking.sh --transcripts alignment_faking_public/data/transcripts.json
#   bash scripts/misalign_behaviors/alignment_faking.sh --run-scenario animal_train_nb --transcripts /path/to/output/

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
TRANSCRIPT_PATH=""           # Path to transcript file(s) or directory (required)
OUTPUT_DIR=""                # Auto-set if empty
SCENARIO=""                  # Python module to run (e.g. animal_train_nb), empty to skip generation
MAX_SAMPLES="150"            # Maximum number of transcripts to convert

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AF_DIR="$PROJECT_DIR/ood_misalignment_eval/alignment_faking_public"

# ── Parse CLI overrides ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --transcripts)   TRANSCRIPT_PATH="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";        shift 2 ;;
        --run-scenario)  SCENARIO="$2";          shift 2 ;;
        --max-samples)   MAX_SAMPLES="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TRANSCRIPT_PATH" ]; then
    echo "Error: --transcripts is required"
    echo ""
    echo "Usage: $0 --transcripts <path_to_transcripts> [--output-dir <dir>] [--run-scenario <module>]"
    echo ""
    echo "The transcript path can be:"
    echo "  - A single JSON file containing conversation transcripts"
    echo "  - A directory containing multiple JSON transcript files"
    echo ""
    echo "Note: alignment_faking_public data is typically stored on Google Drive."
    echo "You may need to download transcripts first."
    exit 1
fi

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

echo "=========================================="
echo "Alignment Faking Pipeline"
echo "=========================================="
echo "Transcripts:  $TRANSCRIPT_PATH"
echo "Scenario:     ${SCENARIO:-skip generation}"
echo "Output:       ${OUTPUT_DIR:-auto}"
echo "=========================================="
echo ""

# ── Step 1: Setup uv venv (if running a scenario) ───────────────────────────
if [ -n "$SCENARIO" ]; then
    if [ ! -d "$AF_DIR/.venv" ]; then
        echo "[Setup] Creating uv virtual environment in alignment_faking_public/..."
        cd "$AF_DIR"
        uv venv
        uv pip install anthropic openai attrs cattrs
        echo "  Done."
        echo ""
    fi

    # ── Step 2: Run scenario ──────────────────────────────────────────────────
    echo "[Step 1/2] Running alignment faking scenario: $SCENARIO..."
    uv run --directory "$AF_DIR" python "model_organism/${SCENARIO}.py"
    echo ""

    echo "[Step 2/2] Converting transcripts to bloom format..."
else
    echo "[Step 1/1] Converting transcripts to bloom format..."
fi

# ── Convert to bloom format ──────────────────────────────────────────────────
CONVERT_CMD="python $SCRIPT_DIR/convert_alignment_faking.py $TRANSCRIPT_PATH"
if [ -n "$OUTPUT_DIR" ]; then
    CONVERT_CMD="$CONVERT_CMD --output_dir $OUTPUT_DIR"
fi
if [ -n "$MAX_SAMPLES" ]; then
    CONVERT_CMD="$CONVERT_CMD --max_samples $MAX_SAMPLES"
fi
eval $CONVERT_CMD

# ── Detect output directory ─────────────────────────────────────────────────
if [ -z "$OUTPUT_DIR" ]; then
    # Infer default output path
    if [ -f "$TRANSCRIPT_PATH" ]; then
        OUTPUT_DIR="$(dirname "$TRANSCRIPT_PATH")/bloom_rollout"
    else
        OUTPUT_DIR="${TRANSCRIPT_PATH}/bloom_rollout"
    fi
fi

echo ""
echo "=========================================="
echo "Done! Output files:"
echo "=========================================="
echo "  $OUTPUT_DIR/rollout.json"
echo ""
echo "Note: No automatic judgment.json is generated for alignment faking."
echo "Alignment faking detection requires manual analysis."
