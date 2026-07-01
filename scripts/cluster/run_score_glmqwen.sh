#!/bin/bash
# Score GLM-4.7-Flash or Qwen3.6-35B-A3B rollouts with their committed
# 18-indicator v2.6 probe ensemble (probes_paper_main / probes_qwen36), on GPU.
# Mirrors run_score.sh, but GLM/Qwen are the PAPER models: they are NOT in
# cross_model_registry and are scored on ALL 5 behaviors x {misaligned, benign}
# (10 dirs), not just the 2 tool-free conversation behaviors the new targets use.
# probe_eval.evaluate reads the base model from each probe's cfg.yaml
# (glm-9b-flash / qwen-35b) -> no per-model scorer code needed.
# Usage: run_score_glmqwen.sh <glm|qwen>
set -euo pipefail

: "${HOME:=/home/jtaraz}"; export HOME   # condor's clean env may not set HOME (set -u)
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

MODEL="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-/fast/jtaraz/huggingface}"
export SOFTFILELOCK=1                 # consumed by scripts/cluster/sitecustomize.py
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
# torch 2.9 reads PYTORCH_CUDA_ALLOC_CONF; keep the legacy name too. expandable
# segments avoids the fragmentation that tipped GLM's lm_head forward into OOM.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO:$REPO/scripts/cluster:${PYTHONPATH:-}"

# HF token (GLM/Qwen download via the node proxy on first load) + any API keys.
set -a; [ -f bloom/.env ] && source bloom/.env; set +a

source /etc/profile.d/modules.sh
module load cuda/12.9
PY="${ENVS_BASE:-/fast/jtaraz/envs}/probe/bin/python"

case "$MODEL" in
  glm)
    PROBE_DIR="probes_paper_main"
    SLUG="glm_4_7_flash"
    # GLM misaligned sycophancy is the legacy bare-named dir (no model slug).
    MIS_SYCO="bloom-results/sycophancy"
    ;;
  qwen)
    PROBE_DIR="probes_qwen36"
    SLUG="qwen3_6_35b_a3b"
    MIS_SYCO="bloom-results/sycophancy_${SLUG}"
    # Qwen-35B full-attention layers OOM under eager attn on the longest agentic
    # rollouts (seq×seq materialization). SDPA is memory-efficient + equivalent.
    export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
    ;;
  *)
    echo "unknown model '$MODEL' (use: glm | qwen)"; exit 1 ;;
esac

# All 5 behaviors x {misaligned, benign}. Misaligned sycophancy handled above.
CANDIDATES=(
  "$MIS_SYCO"
  "bloom-results/sycophancy_benign_${SLUG}"
  "bloom-results/instructed-strategic-sandbagging_${SLUG}"
  "bloom-results/instructed-strategic-sandbagging_benign_${SLUG}"
  "bloom-results/self-preservation_${SLUG}"
  "bloom-results/self-preservation_benign_${SLUG}"
  "bloom-results/strategic-deception_${SLUG}"
  "bloom-results/strategic-deception_benign_${SLUG}"
  "bloom-results/instructed-covert-code-sabotage_${SLUG}"
  "bloom-results/instructed-covert-code-sabotage_benign_${SLUG}"
)

ROLLOUTS=()
for d in "${CANDIDATES[@]}"; do
  if [ -d "$REPO/bloom/$d" ]; then
    ROLLOUTS+=("bloom/$d")
  else
    echo "WARN: missing rollout dir, skipping: bloom/$d"
  fi
done
[ "${#ROLLOUTS[@]}" -gt 0 ] || { echo "no rollout dirs found for $MODEL"; exit 1; }

ALL_PROBES=$(find "probe/${PROBE_DIR}/v4_v2_6_combined_v2_span" -name cfg.yaml -exec dirname {} \; | sort | tr '\n' ' ')
OUT="probe_eval/results_${MODEL}/v4_v2_6_combined_v2_span"

# Fail FAST on a node whose GPU/driver is wedged (observed on i105: torch logs
# "CUDA unknown error -> 0 devices", device_map="auto" then silently loads the
# 35B model onto CPU and extraction crawls for hours). Exit immediately so the
# slot frees and a resubmit lands on a healthy node.
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 7)"; then
    echo "[$(date)] FATAL: CUDA not available on $(hostname) — exiting fast so the slot frees (resubmit to retry on another node)"
    exit 7
fi

# GLM-4.7-Flash (~64GB resident) and Qwen-35B (~70GB) leave little headroom on a
# single 80GB H100; the lm_head forward OOMs at batch_size 4. batch_size 1 (the
# extraction chunk size) keeps the transient small. Override with BATCH=2 etc.
BATCH="${BATCH:-1}"

# --skip_existing skips a rollout when all 36 detector token_scores.json files
# merely EXIST — it does NOT check they hold every dialogue. A crashed/OOM'd run
# leaves partial files that then get silently skipped (we hit this: a batch-4 OOM
# left GLM dirs with 0-16 of 20-40 dialogues, and the rerun skipped them). So
# default to OFF (full clean rescore); opt in only on a known-complete tree with
# SKIP_EXISTING=1 for resume-on-preemption.
SKIP_FLAG=""; [ -n "${SKIP_EXISTING:-}" ] && SKIP_FLAG="--skip_existing"

echo "[$(date)] score $MODEL on $(hostname): $(echo $ALL_PROBES | wc -w) probe dirs x ${#ROLLOUTS[@]} rollout dirs, batch=$BATCH skip=${SKIP_EXISTING:-0} -> $OUT"
printf '  rollout: %s\n' "${ROLLOUTS[@]}"
"$PY" -m probe_eval.evaluate \
    --experiment_folder ${ALL_PROBES} \
    --rollout_dir "${ROLLOUTS[@]}" \
    --output_dir "$OUT" \
    --behavior_threshold 5 --all_negative --batch_size "$BATCH" $SKIP_FLAG
echo "[$(date)] done score $MODEL"
