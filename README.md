# Misalignment Indicator Probes — Pipeline

End-to-end pipeline for training **linear probes on GLM-4.7-Flash activations** that detect misaligned reasoning, evaluating them on rollouts, and computing transcript-TPR / turn-FPR AUROC.

Target model: `zai-org/GLM-4.7-Flash` · Probe layers: 27, 29 · Indicator taxonomy: v2.6 · Label mode: span.

## What's in this repo

The paper-main artifacts are checked in — **you don't have to re-run stages 1 and 2 to reproduce the eval**:

| Artifact | Path | Size | Notes |
|---|---|---:|---|
| Training data (18 indicators) | [probe/data/v4_v2_6_combined_v2/](probe/data/v4_v2_6_combined_v2/) | 173 MB | Output of stage 1 (`probe.generate_v4`) |
| Trained probes (paper-main) | [probe/probes_paper_main/v4_v2_6_combined_v2_span/](probe/probes_paper_main/v4_v2_6_combined_v2_span/) | 1.1 MB | `detector.pt` + `cfg.yaml` + `training_meta.json` for 18 indicators × layers {27, 29} |
| Trained probes (Qwen cross-model) | [probe/probes_qwen36/v4_v2_6_combined_v2_span/](probe/probes_qwen36/v4_v2_6_combined_v2_span/) | 1.1 MB | Same 18 indicators retrained on Qwen3.6-35B-A3B, layers {23, 25} (§8); `val_acts.pt` gitignored |
| OOD eval rollouts | [ood_misalignment_eval/](ood_misalignment_eval/) | ~95 MB | Per-folder `rollout.json` + GT + judge scores for the paper test set |
| Bloom rollouts (committed) | [bloom/bloom-results/](bloom/bloom-results/) | ~200 MB | **20 main dirs** = 5 dev behaviors × {misaligned, benign} × {GLM-4.7-Flash, Qwen3.6-35B-A3B} (dir contents in the layout below). GLM English **val** = authoritative copy of HF (see provenance); Qwen = cross-model set (§8, git-only). Extra exploratory dirs (`instructed-covert-sandbagging*`, `*_qwen3_next`) aren't part of the main set. |
| Bloom config | [bloom/bloom-data/](bloom/bloom-data/) | <1 MB | `behaviors.json`, `models.json`, `seed.yaml`, `behaviors/examples/`, `configurable_prompts/` (input for `bloom run`) |
| Tuned firing thresholds | [probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json](probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json) | <1 MB | Per-probe thresholds at 10 FPR targets |

> **Rollout data provenance.** The Bloom rollouts/transcripts live on HF at [`kzhou35/misalignment-indicators-bloom-rollouts`](https://huggingface.co/datasets/kzhou35/misalignment-indicators-bloom-rollouts). The **English val set is committed** under `bloom/bloom-results/`; everything else is **gitignored** and pulled from HF. Dataset layout:
>
> | HF path | Split / language | Used by |
> |---|---|---|
> | `bloom-results/` | val · English | main eval (committed locally) |
> | `bloom-results-test/` | test · English | main eval |
> | `bloom-results-vi/` | val · Vietnamese | cross-lingual robustness (§7) |
> | `bloom-results-test-vi/` | test · Vietnamese | cross-lingual robustness (§7) |
> | `ood-xlingual/<bench>_vi/` | OOD (deceptionbench, mask, sycophancy_eval) · Vietnamese | cross-lingual robustness (§7) |
>
> Re-upload variants with `python -m bloom.upload_rollouts_hf` (see §7).

The committed `probes_paper_main/` is all you need to reproduce the eval. The full `probe/probes/v4_v2_6_combined_v2_span/` (32 GB, includes `val_acts.pt` activation caches) is **gitignored** and only needed if you retrain or run `recompute_thresholds`.


## Setup

There are **two environments** by role (both created with [`uv`](https://docs.astral.sh/uv/)):

```bash
# (1) Probe stack — GPU. Probe training, eval (GLM forward pass), AUROC, cascade.
#     transformers>=5.0 is REQUIRED: GLM-4.7-Flash is model_type `glm4_moe_lite`,
#     unsupported on transformers 4.x. torch line is the CUDA build (adjust to your CUDA).
uv venv /path/to/envs/probe --python 3.12
uv pip install --python /path/to/envs/probe -r requirements-probe.txt

# (2) Bloom/judge stack — API/CPU. Rollout generation, LLM judges + audit,
#     CoT translation, HF dataset upload. No torch.
uv venv /path/to/envs/bloom --python 3.12
uv pip install --python /path/to/envs/bloom -r requirements-bloom.txt
uv pip install --python /path/to/envs/bloom -e ./bloom    # the bloom package

# Caches (set in every shell/SLURM script so they don't fill node-local disk):
export HF_HOME=/workspace-vast/pretrained_ckpts
```

All commands below are run from the **repo root**. API keys (Anthropic / OpenRouter / HF)
are read from `.env` (and `bloom/.env`); `source` it before running judges/rollouts.

---

## 1. Probe data collection

Generates synthetic transcripts (positive + hard-negative + benign-overtrigger) per indicator, used to train probes. Opus ideates categories + domains per indicator; Sonnet generates the actual transcripts.

```bash
# All indicators in the v2.6 taxonomy
python -m probe.generate_v4 --indicator-set v2_6 --all --k 300 --max-concurrent 10

# Or one indicator at a time
python -m probe.generate_v4 --indicator "Action Concealment" --k 50
```

**Output:** `probe/data/v4_v2_6/<indicator>.json` — list of transcripts with per-turn `indicator_span` labels.

The paper-main training data is at `probe/data/v4_v2_6_combined_v2/` (~173 MB, 18 indicators).

---

## 2. Probe training

Trains a logistic-regression probe per `(indicator, layer)` on GLM activations from the synthetic transcripts. Tokens inside indicator spans are positive; all other assistant tokens are negative.

```bash
# Single SLURM job (paper-main config)
bash scripts/probe_train_eval/run_v2_6_v4_train.sh \
    --label-mode span --detect-layers "27 29"

# Or directly (one indicator)
python -m probe.train --indicator "Action Concealment" \
    --label-mode span --detect-layers 27 29 --reg-coeff 10.0
```

`train.py` skips indicators whose probes already exist, so re-running fills gaps incrementally.

**Output:** `probe/probes/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/`
- `detector.pt` — probe weights (~26 KB)
- `cfg.yaml` — model name, layers, hyperparams
- `training_meta.json` — val thresholds at the tuned FPR targets
- `val_acts.pt` — cached validation activations (large, regeneratable)

**Recompute thresholds only** (after retraining with new metric defs):
```bash
python -m probe.recompute_thresholds \
    --probe-dir probe/probes/v4_v2_6_combined_v2_span \
    --data-dir probe/data/v4_v2_6_combined_v2
```

---

## 3. Probe eval (score rollouts)

For each `(probe, rollout_dir)` pair, runs GLM-4.7-Flash forward pass, extracts activations at layer 27/29, applies the probe, and writes per-token scores.

```bash
# All paper dev + test behaviors via SLURM (2 parallel jobs)
bash scripts/probe_train_eval/run_v2_6_v4_eval.sh --label-mode span --batch-size 4

# Or a single batched invocation with all 36 probes × N rollout dirs.
# Point at probes_paper_main/ (committed, lightweight) when reproducing.
ALL_PROBES=$(find probe/probes_paper_main/v4_v2_6_combined_v2_span -name cfg.yaml \
    -exec dirname {} \; | sort | tr '\n' ' ')
python -m probe_eval.evaluate \
    --experiment_folder ${ALL_PROBES} \
    --rollout_dir bloom/bloom-results/strategic-deception_glm_4_7_flash \
                  bloom/bloom-results/sycophancy \
    --behavior_threshold 5 --all_negative --batch_size 4
```

**Output:** `probe_eval/results/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/<rollout_folder_basename>/`
- `results.json` — per-rollout & per-turn probe scores + AUROC
- `token_scores.json` — per-token raw probe scores (used by downstream cascade + AUROC scripts)
- `visualization.html` — token-highlighted transcript browser

**Notes:**
- Output path is keyed on `Path(rollout_dir).name`. If you re-evaluate the same folder under a different condition (e.g. translated rollouts), use a suffixed symlink or rename — otherwise results overwrite.
- `--reasoning_only` restricts probe scoring to `<think>` tokens only.

---

## 4. Tune per-probe firing thresholds

After eval, recompute per-probe thresholds on the held-out neutral set so that each probe fires at a target turn-level FPR (default sweep: 0.002 → 0.3).

```bash
# Score neutral dialogues with each probe, save sentence-level scores
bash scripts/probe_train_eval/run_score_neutral_slurm.sh

# Tune thresholds from sentence scores at multiple FPR targets
python -m probe.recompute_thresholds  # (or use the score-neutral skill)
```

**Output:** `probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json` — keyed by `fpr_<target>_swp` and `layer{27,29}`, giving per-probe scalar thresholds.

---

## 5. Compute AUROC (paper convention)

The paper's reporting convention is **transcript-level TPR × turn-level FPR**: every GT-positive transcript contributes one positive (max sentence-score across turns); every benign-folder turn + non-misaligned turn in pos transcripts contributes a negative. Ambiguous + benign-audit-suspicious transcripts are excluded.

Implemented in [probe_eval/paired_bootstrap_ci.py](probe_eval/paired_bootstrap_ci.py):

```bash
# Point estimates + paired bootstrap CIs across methods
python -m probe_eval.paired_bootstrap_ci --n-replicates 1000 --seed 0
```

The script reports AUROC for:
- **Probes**: OR-fused across 18 indicators at each tuned FPR target (discrete ROC).
- **Probes + Haiku cascade**: probe stage-1 flags → Haiku 4.5 stage-2 verifier (see [probe_eval/probe_llm_cascade.py](probe_eval/probe_llm_cascade.py)).
- **LLM-judge baselines** (Haiku, Qwen3, each with / without indicator taxonomy) — continuous ROC.

This produces the pooled AUROCs reported as the paper's main result (`tab:main-auroc`).

Quick standalone AUROC check (no bootstrap):

```bash
python -m probe_eval.print_auroc_grid \
    --results-subdir v4_v2_6_combined_v2_span \
    --thresholds probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json
```

---

## 6. Cascade (probes + Haiku 4.5)

Stage-1 probe flags → Stage-2 Haiku-4.5 verifier on flagged turns only. Final positive iff both fire.

```bash
python -m probe_eval.probe_llm_cascade \
    --results-subdir v4_v2_6_combined_v2_span \
    --thresholds probe_eval/results/v4_v2_6_combined_v2_span/tuned_thresholds.json \
    --judge-model claude-haiku-4-5-20251001 --concurrency 30
```

Verdicts are cached per `(behavior, var, rep, turn)` so resumption is cheap.

---

## 7. Cross-lingual robustness experiment

Translates the target model's chain-of-thought into another language (Chinese / Vietnamese)
and compares how the Haiku judge vs. the probe ensemble degrade — the §"Cross-Lingual
Robustness" result. Full runbook (translation → repair → judge → probe eval → AUROC →
translation-faithfulness audit), environments, and data layout:

**→ [bloom/README_cross_lingual.md](bloom/README_cross_lingual.md)**

(Agent-facing version with environment gotchas: `.claude/skills/cross-lingual-probe-eval/SKILL.md`.)

(Re)publish the translated Vietnamese rollouts to HF (data locations in the provenance table above):

```bash
python -m bloom.upload_rollouts_hf            # vi: val + test + ood-xlingual
python -m bloom.upload_rollouts_hf --dry-run  # show plan only
```

---

## 8. Evaluate a new target model (cross-model rollouts)

Score a different target on the same 5 dev behaviors with the **scenarios held fixed** (only the
model differs) by reusing the GLM data and re-running just rollout + judge. Done here for
**Qwen3.6-35B-A3B** (`*_qwen3_6_35b_a3b`). Steps 1–4 use the **bloom** env, steps 5–6 the **probe** env.

1. **Register** the model in `bloom/bloom-data/models.json` (OpenRouter id).
2. **Reuse scenarios**: copy the GLM run's `understanding.json` + `ideation.json` into a new
   `bloom/bloom-results/<behavior>_<model>/` (model-agnostic, so prompts/memos stay identical). For
   the **benign** control, copy the `_benign` GLM scenarios and set `behavior.name: <behavior>_benign`,
   `configurable_prompts: "default"` (no memo/sandbag injection).
3. **Rollout**: `bloom rollout <config_dir>` with a behavior seed at `rollout.target: <model>`. Set
   `evaluator_reasoning_effort: none` — a thinking-mode Opus-4.6 evaluator rejects the assistant-prefill
   bloom hands it on reasoning-only target turns and truncates the rollout (the target's reasoning stays on).
4. **Ground truth + audit** (Opus-4.6, as for GLM):
   ```bash
   # misaligned dir: open-ended judge, then indicator judge (merges into the same file)
   python -m black_box_ind_judge <dir>/rollout.json --misalignment-turns \
       --model claude-opus-4-6 --thinking-budget 20000 -o <dir>/rollout_misalignment_turns.json
   python -m black_box_ind_judge <dir>/rollout.json --indicator-turns \
       --test-behavior <behavior> --thinking-budget 20000 -o <dir>/rollout_misalignment_turns.json
   # benign dir: indicator judge in benign mode only — add `--indicator-turns --benign`,
   #   with --test-behavior set to the base behavior (e.g. sabotage, sandbagging)
   python -m black_box_ind_judge.audit --source bloom-results/<behavior>_<model> --audit-benign
   ```
5. **Train Qwen probes** at depth-matched layers 23/25 (≈60% depth, GLM's 27/29 analogue):
   `bash scripts/probe_train_eval/run_qwen36_train.sh` (= `probe.train --model qwen-35b --detect-layers 23 25`)
   → `probe/probes_qwen36/v4_v2_6_combined_v2_span/<indicator>/span/layer{23,25}/`.
6. **Score + AUROC** with the Qwen per-turn scorer. Qwen keeps `<think>` only on the *final* turn, so each
   turn is scored with the dialogue truncated to end there; multi-step tool turns are reduced by **max over
   steps** (matches the GLM scorer — `--step-reduce last` reproduces the old last-step behavior):
   ```bash
   python bloom/score_auroc_qwen_perturn.py --layers 23 25 --step-reduce max   # dev
   python bloom/score_ood_and_merge_qwen.py                                    # all-9 (test 5 + 4 OOD; reads the committed test scores)
   ```
   `recompute_qwen_auroc.py` recomputes AUROC from the cached scores on CPU.

---

## File layout summary

Legend: ✓ committed to git · ✗ gitignored (regeneratable)

```
misalignment-indicators/          # repo root (run all commands from here)
├── probe/
│   ├── generate_v4.py             # data gen                         ✓
│   ├── train.py                   # probe training                   ✓
│   ├── recompute_thresholds.py    # threshold tuning                 ✓
│   ├── data/v4_v2_6_combined_v2/  # training data (18 indicators)    ✓ 173 MB
│   ├── probes_paper_main/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/
│   │                              # paper probes (lightweight, no val_acts)  ✓ 1.1 MB
│   ├── probes_qwen36/v4_v2_6_combined_v2_span/<indicator>/span/layer{23,25}/
│   │                              # Qwen cross-model probes, lightweight (§8)  ✓ 1.1 MB
│   └── probes/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/
│                                  # full probes with val_acts.pt cache  ✗ 32 GB
├── probe_eval/
│   ├── evaluate.py                # run probes on rollouts           ✓
│   ├── paired_bootstrap_ci.py     # AUROC + paired CI                ✓
│   ├── probe_llm_cascade.py       # 2-stage cascade                  ✓
│   ├── reasoning_vs_output_metrics.py                                ✓
│   └── results/v4_v2_6_combined_v2_span/
│       ├── tuned_thresholds.json                                     ✓
│       └── <indicator>/span/layer{27,29}/<behavior>/{results,token_scores}.json
│                                                                     ✗ ~35 GB
├── bloom/bloom-data/                      # bloom run config: behaviors.json, models.json, seed.yaml, examples  ✓
├── bloom/bloom-results/<behavior>_<model>/  # 20 main dirs: {GLM val (from HF), Qwen cross-model} ✓ ~200 MB
│   ├── rollout.json                       # rollouts (system/user/assistant + reasoning)
│   ├── transcript_v<N>r<M>.json           # per-rollout transcripts
│   ├── rollout_misalignment_turns.json    # Opus GT misaligned turns
│   ├── ambiguous_rollouts.json            # GT audit exclusions
│   └── rollout_misalignment_turns_{haiku,qwen}_*_score.json  # LLM-judge per-turn scores
├── bloom/bloom-results-test/<behavior>/   # test rollouts — pull from HF                         ✗ on HF
├── bloom/README_cross_lingual.md          # cross-lingual experiment runbook ✓
├── bloom/upload_rollouts_hf.py            # push language variants to the HF dataset  ✓
├── bloom/{translate_reasoning,run_haiku_judge,compute_*,check_label}*  # cross-lingual scripts  ✓
├── bloom/{score_auroc_qwen_perturn,score_ood_and_merge_qwen,recompute_qwen_auroc,gen_qwen_test,gt_qwen_test}.py  # Qwen cross-model (§8)  ✓
├── bloom/bloom-results-vi, bloom-results-test-vi, ood-xlingual/*_vi  # VI translations  ✗ on HF
├── bloom/bloom-results-cn, bloom-results-test-cn, ood-xlingual/*_cn  # CN translations  ✗ local only
├── ood_misalignment_eval/<dataset>/...    # OOD test set rollouts + GT + judge scores  ✓ 95 MB
└── scripts/probe_train_eval/
    ├── run_v2_6_v4_train.sh                                          ✓
    ├── run_v2_6_v4_eval.sh                                           ✓
    ├── run_score_neutral_slurm.sh                                    ✓
    └── run_qwen36_{train,score,score_perturn}.sh   # Qwen cross-model (§8)  ✓
```

**To reproduce paper eval from scratch, you only need stages 3–6** — training data and trained probes are committed.
