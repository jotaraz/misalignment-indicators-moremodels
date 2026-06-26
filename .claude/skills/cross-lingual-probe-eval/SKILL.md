---
name: cross-lingual-probe-eval
description: Run the indicator-probe evaluation on bloom rollouts (any language / split) to produce probe token scores on GPU and then transcript-TPR/turn-FPR AUROC. Use when asked to "get probe scores", "run probe eval", "compute probe AUROC", reproduce the cross-lingual (EN/CN/VI) probe results, or score a new bloom-results dir with the 18-indicator probe ensemble. Covers the companion Haiku-judge scoring too.
---

# Cross-lingual probe evaluation (probe scores → AUROC)

This is the runbook for the cross-lingual ablation in `paper/6_appendix.tex`
(`app:cross-lingual`): score GLM-4.7-Flash reasoning with the 18-indicator probe
ensemble and with the Haiku judge, across EN / CN / VI, then compute
transcript-TPR / turn-FPR AUROC. The same steps apply to any bloom-results dir
(val or test, any language).

Repo root: `/workspace-vast/kaiwen/misalignment-indicators` (referred to as `$REPO`).

## Pipeline overview

Three stages. The probe path is the GPU one; the Haiku judge is API-only.

1. **Probe token scores** — GPU/SLURM. `probe_eval.evaluate` loads GLM-4.7-Flash
   once, extracts activations per rollout dir, scores all 36 detectors
   (18 indicators × layers 27/29), writes `token_scores.json` per (probe, dir).
2. **Probe AUROC** — CPU. `bloom/compute_probe_auroc.py --split {dev,test,pool}`
   OR-fuses the 18 indicators per turn and computes AUROC (layers 27/29, full +
   reasoning-only). Reasoning-only is derived **post-hoc** from the same token
   scores (no second GPU run).
3. **Haiku judge + AUROC** — API + CPU. `bloom/run_haiku_judge_*.sh` →
   `bloom/compute_haiku_auroc.py --split {dev,test,pool}`.

## Canonical scripts (unified; `--split`/`--preset` select dev / test / ood / pool)

- Translate + repair:          `bloom/translate_reasoning.py --lang {cn,vi} --preset {dev,test,ood} [--repair] [--dry-run]`
- Haiku judge:                 `bloom/run_haiku_judge_test.sh`, `bloom/run_haiku_judge_ood_xlingual.sh`
- Probe eval SLURM generator:  `scripts/probe_train_eval/run_probe_eval_test_three_lang.sh`, `..._ood_xlingual.sh`
- Probe AUROC:                 `bloom/compute_probe_auroc.py --split {dev,test,pool}`
- Haiku AUROC:                 `bloom/compute_haiku_auroc.py --split {dev,test,pool}`
- Faithfulness audit:          `bloom/audit_faithfulness.py {vi,en,crosstab}`

See `bloom/README_cross_lingual.md` for the human-facing runbook. `pool` = the 10 test
Bloom folders + the 3 OOD benchmarks (the paper's full test set).

## The 5 cross-lingual behaviors (10 folders = ×{benign, misaligned})

`instructed-covert-code-sabotage`, `instructed-strategic-sandbagging`,
`self-preservation`, `strategic-deception`, `sycophancy`. This matches
`bloom/bloom-results-cn` exactly. The test set has 8 extra folders
(`long-horizon-sabotage-{backdoor,bug}`, `self-preferential-bias`,
`self-promotion`) — **exclude them**; they are not part of this ablation and
several lack the `rollout_misalignment_turns.json` GT.

## Environment

The probe-eval venv is `/workspace-vast/kaiwen/envs/dd`. The hard-won facts
(June 2026 — burned ~hours of debugging, do not relearn these):

- **Venv MUST be on shared `/workspace-vast`**, NOT `/home/$USER` (per-node local
  disk — a SLURM job on another node gets `python: No such file or directory`).
- **`uv.lock` is STALE / WRONG for GLM-4.7-Flash.** It pins transformers 4.57.6 +
  torch 2.2.2, but GLM-4.7-Flash is `Glm4MoeLiteForCausalLM` (`model_type:
  glm4_moe_lite`) with NO bundled remote code, and **`glm4_moe_lite` first ships
  in transformers 5.0.0** — no 4.x can load it. So do NOT `uv sync --frozen`.
  Install: **transformers==5.9.0, torch==2.9.0, huggingface-hub>=1.5,<2,
  tokenizers>=0.22,<=0.23** (torchvision not needed — not imported by probe eval).
  transformers 5.x + torch 2.9 import deception_detection/probe_eval and load the
  detectors with no API breakage.
- **NFS install trap:** `UV_LINK_MODE=copy` over the `/workspace-vast` NFS leaves
  PARTIAL package installs and dies with `failed to remove directory .../__pycache__:
  Directory not empty (os error 39)` (symptom: `torch.__file__ is None`,
  `module 'torch' has no attribute '__version__'`). Fix: use
  **`UV_LINK_MODE=hardlink`** (cache + venv are co-located on `/workspace-vast`, so
  hardlink is atomic), and when reinstalling a package first `rm -rf` its dir +
  dist-info so the install is a pure copy with no removal step:
  ```bash
  cd $REPO/ood_misalignment_eval/deception-detection
  export UV_CACHE_DIR=/workspace-vast/kaiwen/.cache/uv UV_LINK_MODE=hardlink
  uv pip install --python /workspace-vast/kaiwen/envs/dd \
    transformers==5.9.0 torch==2.9.0 'huggingface-hub>=1.5,<2' 'tokenizers>=0.22,<=0.23'
  # if a package half-installs: rm -rf its dir in site-packages, then reinstall.
  ```
  (zsh aborts a whole `rm` line on a non-matching glob like `torch-*`; use exact
  names or `find -maxdepth 1 -name '...' -exec rm -rf {} +`.)
- **Two gitignored data files are missing** (the `data/` dir is empty since
  `persona_vectors` was deleted) and BOTH are read at import / Experiment-construct
  but UNUSED by probe scoring — create them so the chain doesn't crash:
  - `data/black_box_baseline/prompt.txt` — any one-line placeholder.
  - `data/repe/true_false_facts.csv` — columns `statement,label`; the probe's
    `train_data` (`repe_honesty...`) is constructed at `Experiment.__init__` but
    never used at eval (`get_detector` loads the saved `detector.pt`), so synthetic
    rows (label 1 = true, >5 words each) give byte-identical eval results.
- Model: `glm-9b-flash` → `zai-org/GLM-4.7-Flash`, cached under
  `/workspace-vast/pretrained_ckpts/hub`. Set `HF_HOME=/workspace-vast/pretrained_ckpts`.
  Loads ~751 weight shards (~3 min) — slow but normal.
- Probe set: `probe/probes_paper_main/v4_v2_6_combined_v2_span` (36 `cfg.yaml`).
- **SLURM: use `qos=high`.** Both `low` AND `normal` get preempted by other users'
  `qos=high` jobs (observed: `normal` jobs killed mid-run with `PreemptTime` set,
  on a busy cluster, within minutes). `high` is the only non-preempted tier
  (16 GPU/user cap; 1 GPU here is fine). The batch script runs past the killed
  `srun` so a preempted job does NOT auto-requeue — another reason to use `high`.
  Note: the cgroup write errors (`Unable to move pid to init root cgroup`) in every
  job log are harmless cleanup noise, NOT the failure cause.
- AUROC scripts: numpy in the `dd` venv is 1.26 (`np.trapezoid` is 2.0+); the
  `*_test.py` scripts fall back to `np.trapz`. Run them with the `dd` venv (`hfdl`
  lacks numpy). Haiku judge venv is `/workspace-vast/kaiwen/envs/hfdl` (has
  `anthropic`); source `$REPO/.env` for `ANTHROPIC_API_KEY`.

## Stage 1 — probe token scores (GPU)

The output dir for each (probe, rollout) is
`probe_eval/results/<probe_set>/<indicator>/span/layer{27,29}/<rollout_basename>/token_scores.json`
(via `relative_to(parents[3])` in `evaluate.py`; works with `probes_paper_main`).
The AUROC scripts key on `<rollout_basename>`, so it must carry the language:

- EN: pass `bloom/bloom-results-test/test_<behavior>` directly (basename `test_<behavior>`).
- CN/VI: create a symlink dir so basenames get a `_cn`/`_vi` suffix:
  ```bash
  mkdir -p $REPO/bloom/bloom-results-test-langs && cd $_
  for d in ../bloom-results-test-cn/test_*/; do b=$(basename "$d"); ln -sfn "../bloom-results-test-cn/$b" "${b}_cn"; done
  for d in ../bloom-results-test-vi/test_*/; do b=$(basename "$d"); ln -sfn "../bloom-results-test-vi/$b" "${b}_vi"; done
  ```

Run:
```bash
# dry-run prints the generated sbatch; --submit submits it
bash scripts/probe_train_eval/run_probe_eval_test_three_lang.sh --submit
```
Key flags in the job: `--all_negative` (skips judgment.json, labels everything
negative — GT comes separately at AUROC time), `--skip_existing` (resumable on
preemption), `--batch_size 4`. SLURM: `qos=low` (free, preemptible) + `--requeue`,
`gres=gpu:1`, `mem=128G`, `time=16:00:00`, all cluster rules (no `--export`,
`srun`-wrapped, cleanup trap, `NCCL_SOCKET_IFNAME=vxlan0`).

**Always sanity-run 2 dirs first** (one EN + one `_cn`) to validate the GLM load
and both path types before the full sweep. Verify it wrote
`token_scores.json` for both dirs and both layers, then submit the full job.

## Stage 2 — probe AUROC (CPU)

```bash
# probe venv; prints layers 27/29, full + reasoning-only
/workspace-vast/kaiwen/envs/dd/bin/python bloom/compute_probe_auroc.py --split test   # or pool
```
Reasoning-only re-partitions each turn's tokens via `_partition_turn` from the
already-written token scores — no extra GPU run. AUROC convention: pos =
max-over-turns max-over-sentences max-over-probes on transcripts with a
GT-misaligned turn; neg = every non-misaligned turn + all benign-folder turns;
exclude ambiguous/suspicious via `load_unreliable_rollouts`.

## Stage 3 — Haiku judge + AUROC (API + CPU)

```bash
bash bloom/run_haiku_judge_test.sh all   # cn | vi | en | all
/workspace-vast/kaiwen/envs/dd/bin/python bloom/compute_haiku_auroc.py --split test   # or pool
```
Produces 4 files per folder/lang: `{nodef,indtax}` × `{full, RONLY}`
(`rollout_misalignment_turns_haiku_history_<suffix>_score.json`). `nodef` =
`--no-definition`, `indtax` = `--use-indicator-taxonomy`, RONLY = `--no-response`.

**Critical gotcha:** the translation scripts copy every non-`rollout.json` file
(including the EN `*_haiku_history_{nodef,indtax}_score.json`) into the CN/VI
dirs. A skip-if-exists guard would then keep EN scores. `run_haiku_judge_test.sh`
**force-overwrites** CN/VI; EN only generates the two RONLY files it lacks.

## Output locations

- Probe token scores: `probe_eval/results/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/<basename>/token_scores.json`
- Haiku scores: next to each `rollout.json`
- AUROC tables: stdout of the four `compute_*_test.py` scripts
