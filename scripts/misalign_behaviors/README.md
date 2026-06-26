# Misalignment Behavior Evaluation Scripts

Scripts to run LLM evaluations on misalignment behavior datasets and convert results to a standardized **bloom format** (`rollout.json` + `judgment.json`).

## Datasets

| Script | Dataset | Description |
|--------|---------|-------------|
| `bloom.sh` | [BLOOM](../../bloom/) | Internal behavior eval framework (understanding → ideation → rollout → judgment) |
| `deception_detection.sh` | [deception-detection](../../ood_misalignment_eval/deception-detection/) | Insider trading, roleplaying, sandbagging, undermining oversight scenarios |
| `deception_bench.sh` | [DeceptionBench](../../ood_misalignment_eval/deception-bench/) | Sycophantic misrepresentation, alignment faking, sandbagging |
| `agentic_misalignment.sh` | [agentic-misalignment](../../ood_misalignment_eval/agentic-misalignment/) | Agentic misalignment in email/urgency-based tasks |
| `alignment_faking.sh` | [alignment_faking_public](../../ood_misalignment_eval/alignment_faking_public/) | Alignment faking transcripts with reasoning scratchpads |
| `impossiblebench.sh` | [ImpossibleBench](../../ood_misalignment_eval/impossiblebench/) | Impossible coding tasks (LCB/SWE) — original, oneoff, conflicting splits |
| `sycophancy_eval.sh` | [sycophancy-eval](../../ood_misalignment_eval/sycophancy-eval/) | Sycophancy evaluation (answer, are_you_sure, feedback) |

Each shell script has a corresponding `convert_*.py` for format conversion (except bloom, which handles conversion internally).

## Setup

### 1. Environment Variables

Create a `.env` file at the repo root (`data_generation/.env`) with your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...    # required for deception_bench, sycophancy_eval
HF_TOKEN=hf_...                  # optional, for private HF models
```

### 2. Per-Dataset Dependencies

Each dataset has its own environment. Scripts auto-setup where possible.

| Dataset | Python | Package Manager | Key Dependencies |
|---------|--------|-----------------|------------------|
| bloom | 3.11+ | pip | `litellm`, `pyyaml`, `wandb` |
| deception-detection | 3.11+ | pip/venv | `torch~=2.2.0`, `transformers`, `anthropic`, `together` |
| deception-bench | 3.10+ | pip | `inspect_ai>=0.3.0`, `anthropic`, `litellm` |
| agentic-misalignment | 3.10+ | uv | `anthropic`, `openai`, `google-generativeai`, `together` |
| alignment_faking_public | 3.10+ | uv | `anthropic`, `openai` |
| impossiblebench | 3.10+ | uv | `inspect_ai>=0.3.0`, `swebench>=4.0.0`, `pandas` |
| sycophancy-eval | 3.10+ | pip | `openai` (via OpenRouter) |

Scripts using `uv` will create virtual environments automatically on first run.

## Usage

### Run a full pipeline

```bash
# BLOOM behavioral eval
bash scripts/misalign_behaviors/bloom.sh --stage rollout --debug

# Deception detection (local GPU or API)
bash scripts/misalign_behaviors/deception_detection.sh \
  --dataset roleplaying__plain --model glm-9b-flash --use-api False

# DeceptionBench via OpenRouter
bash scripts/misalign_behaviors/deception_bench.sh \
  --model z-ai/glm-4.7-flash --type Sandbagging --judge

# Agentic misalignment
bash scripts/misalign_behaviors/agentic_misalignment.sh \
  --config configs/example_experiment_config.yaml

# ImpossibleBench
bash scripts/misalign_behaviors/impossiblebench.sh \
  --task replicate_lcb --split oneoff

# Sycophancy eval
bash scripts/misalign_behaviors/sycophancy_eval.sh \
  --model glm-4.7-flash --dataset all
```

### Convert existing results only

```bash
# Skip inference, just convert to bloom format
bash scripts/misalign_behaviors/deception_bench.sh --skip-inference
bash scripts/misalign_behaviors/impossiblebench.sh --skip-generation

# Or run converters directly
python scripts/misalign_behaviors/convert_agentic_misalignment.py <results_dir>
python scripts/misalign_behaviors/convert_deception_bench.py <jsonl_file>
python scripts/misalign_behaviors/convert_impossiblebench.py <log_dir> --split oneoff
python scripts/misalign_behaviors/convert_alignment_faking.py <transcripts_path>
```

## Output Format

All scripts produce standardized bloom-format output:

- **`rollout.json`** — Transcripts with system/user/assistant messages as structured events
- **`judgment.json`** — Per-rollout `behavior_presence` scores (1–10 scale; 10 = strong misalignment signal)

Output locations vary by dataset (see each script's `--help` or defaults in the script header).
