#!/bin/bash
# Run bloom pipeline to generate rollouts and judgments.
#
# Output:
#   bloom/bloom-results/<behavior>/rollout.json + judgment.json
#
# Usage:
#   bash scripts/misalign_behaviors/bloom.sh                    # Run full pipeline
#   bash scripts/misalign_behaviors/bloom.sh --stage rollout    # Run rollout stage only
#   bash scripts/misalign_behaviors/bloom.sh --debug

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
STAGE=""                      # Empty = full pipeline. Options: understanding, ideation, rollout, judgment
DEBUG=""

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLOOM_DIR="$PROJECT_DIR/bloom"

# ── Parse CLI arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --debug) DEBUG="--debug"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ── Run bloom ─────────────────────────────────────────────────────────────────
cd "$BLOOM_DIR"

if [ -n "$STAGE" ]; then
    echo "Running: bloom $STAGE bloom-data $DEBUG"
    bloom "$STAGE" bloom-data $DEBUG
else
    echo "Running: bloom run bloom-data $DEBUG"
    bloom run bloom-data $DEBUG
fi
