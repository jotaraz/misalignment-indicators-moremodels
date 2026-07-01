#!/bin/bash
# Generate on-policy "neutral" completions (WildChat/LMSYS/FLAN prompts) and extract
# per-layer activations for ONE model on GPU. HF-generate backend (vLLM isn't
# installed and doesn't support glm4_moe_lite / Qwen3.6 hybrid; the HF model we load
# for extraction generates the completions too). Short seqs (<=1280 tok) so no OOM.
# Usage: run_neutral_extract.sh <glm|qwen>
set -euo pipefail

: "${HOME:=/home/jtaraz}"; export HOME
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

MODEL="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"

export HF_HOME="${HF_HOME:-/fast/jtaraz/huggingface}"
export SOFTFILELOCK=1
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO:$REPO/scripts/cluster:${PYTHONPATH:-}"
set -a; [ -f bloom/.env ] && source bloom/.env; set +a   # HF token for weight download

source /etc/profile.d/modules.sh; module load cuda/12.9
PY="${ENVS_BASE:-/fast/jtaraz/envs}/probe/bin/python"

# Qwen-35B: use SDPA (eager full-attention can blow up; harmless for short seqs but safer/faster)
[ "$MODEL" = qwen ] && export ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# fail fast on a wedged-GPU node (observed: i105 -> CUDA-unknown -> CPU)
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 7)"; then
    echo "[$(date)] FATAL: CUDA not available on $(hostname) — exiting fast (resubmit)"; exit 7
fi

echo "[$(date)] neutral extract: model=$MODEL on $(hostname) (max_prompts=${MAX_PROMPTS:-2000})"
"$PY" -m probe.neutral.extract_activations \
    --model "$MODEL" --gen-backend hf \
    --max-prompts "${MAX_PROMPTS:-2000}" \
    --gen-batch-size "${GEN_BS:-8}" \
    --extract-batch-size "${EXTRACT_BS:-16}" \
    --max-tokens-per-layer "${MAX_TOK_PER_LAYER:-100000}"
echo "[$(date)] done neutral extract: $MODEL -> probe/data/neutral/$MODEL/activations/"
