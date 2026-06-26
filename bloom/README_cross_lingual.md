# Cross-lingual monitoring experiment

Scripts for the cross-lingual ablation (paper §"Cross-Lingual Robustness"): translate
the target model's chain-of-thought into another language (Chinese / Vietnamese),
then compare how the **Haiku LLM judge** and the **18-indicator probe ensemble**
degrade. Headline result: the probe degrades far less than the judge on Vietnamese
(a low-resource language the judge reads poorly), especially in the reasoning-only view.

Everything below lives in `bloom/` unless noted. SLURM probe-eval scripts are in
`scripts/probe_train_eval/`. For a Claude-agent-oriented version of this runbook with
all the environment gotchas, see `.claude/skills/cross-lingual-probe-eval/SKILL.md`.

---

## Environments

Two Python environments are used (activate the relevant one before each step):

- **Probe venv** — probe eval, AUROC, faithfulness audit. This is the `deception-detection`
  venv from the main README's Setup, **upgraded to transformers ≥ 5.0 with a matching torch**:
  GLM-4.7-Flash is `glm4_moe_lite`, which transformers 4.x cannot load, so do **not**
  `uv sync --frozen` (the committed lock pins 4.57.6). Also needs `numpy` and `anthropic`.
- **API venv** — translation, repair, Haiku judge. Any venv with `anthropic` installed.

Notes:
- **API key**: `source .env` from the repo root (sets `ANTHROPIC_API_KEY`).
- Set `HF_HOME` to your shared model cache (same as the main README's Setup).
- Two gitignored data files are read at import (unused by probe eval) — create them if missing:
  `ood_misalignment_eval/deception-detection/data/black_box_baseline/prompt.txt` (any placeholder)
  and `.../data/repe/true_false_facts.csv` (synthetic is fine — `train_data` is constructed but
  never used at eval, since `detector.pt` is loaded).
- Run python scripts from the repo root with `PYTHONPATH=.` so `probe_eval`/`bloom` import.
- For SLURM, the probe venv and its uv cache should be on **shared storage** (not a node-local
  home dir), or a job on another node fails with `python: No such file or directory`.
- The shell / SLURM scripts (`run_haiku_judge_*.sh`, `scripts/probe_train_eval/run_probe_eval_*.sh`)
  hardcode the original run's absolute venv/repo paths — edit those to your paths before running.

---

## Data layout

| dir | contents |
|---|---|
| `bloom-results/`, `bloom-results-test/` | EN rollouts + GT (`rollout_misalignment_turns.json`) + judge scores. Dev = 10 folders; test = `test_*` folders. |
| `bloom-results-{cn,vi}/` | dev translations (10 folders) |
| `bloom-results-test-{cn,vi}/` | test translations (18 `test_*` folders; only 10 used in the ablation — the 5 cross-lingual behaviors × {benign, misaligned}) |
| `bloom-results-test-langs/` | symlinks `test_<b>_{cn,vi}` → the test translation dirs, so probe-eval output dirs carry a language suffix |
| `ood-xlingual/` | the 3 OOD benchmarks with CoT: `deceptionbench`, `mask`, `sycophancy_eval` (EN symlinks) and `<name>_{cn,vi}` translations |
| `probe_eval/results/v4_v2_6_combined_v2_span/<indicator>/span/layer{27,29}/<basename>/token_scores.json` | probe token scores (the AUROC scripts read these) |

The **5 cross-lingual behaviors** (match `bloom-results-cn`): covert-code-sabotage,
strategic-sandbagging, self-preservation, strategic-deception, sycophancy.
Agentic Misalignment is excluded (its rollouts have no separable CoT to translate).

---

## Pipeline (run order)

### 1. Translate reasoning → CN / VI  (+ repair)
`translate_reasoning.py` handles both the first pass and the refusal/derailment repair.
Only `reasoning` blocks are translated (everything else copied verbatim); CN = Sonnet 4.6,
VI = Opus 4.7, with a forensic prompt + inline target-script validation so refusals and
derailments are caught and retried in-pass. `--preset` is `dev` (→ `bloom-results-{cn,vi}`),
`test` (→ `bloom-results-test-{cn,vi}`), or `ood` (→ `ood-xlingual/<name>_{cn,vi}`).

```bash
source .env                                                            # API venv active; from repo root
python bloom/translate_reasoning.py --lang vi --preset test           # full pass (dev|test|ood)
python bloom/translate_reasoning.py --lang cn --preset ood --dry-run  # preview 5 samples
python bloom/translate_reasoning.py --lang vi --preset dev  --repair  # re-do only blocks missing target script
# arbitrary single folder:  --src <EN dir> --dst <out dir>
```
Stubborn blocks Opus refuses entirely are left as the English source (a handful of cases).

### 2. Haiku judge (CN/VI all 4 variants + EN reasoning-only)
Produces `rollout_misalignment_turns_haiku_history_{nodef,indtax}{,_RONLY}_score.json`.
CN/VI are force-overwritten (translation copies the EN score files in); EN only gets the
two reasoning-only (`RONLY`) files it lacks.

```bash
bash bloom/run_haiku_judge_test.sh            # test: cn|vi|en|all (default all)
bash bloom/run_haiku_judge_ood_xlingual.sh    # the 3 OOD benchmarks
# pre-existing dev judges (stale persona_vectors path — fix before reuse):
#   run_haiku_judge_on_cn.sh, run_haiku_judge_on_vi.sh, run_haiku_judge_reasoning_only.sh
```

### 3. Probe token scores (GPU / SLURM)
One model load; 36 detectors (18 indicators × layers 27/29) over all rollout dirs.
`--all_negative` (labels come later from GT), `--skip_existing` (resumable). **Use `qos=high`**
— `normal`/`low` get preempted by other users' high-qos jobs. Sanity-run a couple of dirs
first. Token scores land under `probe_eval/results/...` keyed by rollout-dir basename.

```bash
bash scripts/probe_train_eval/run_probe_eval_test_three_lang.sh --submit   # 10 folders × EN/CN/VI = 30 dirs
bash scripts/probe_train_eval/run_probe_eval_ood_xlingual.sh   --submit   # 3 OOD × EN/CN/VI = 9 dirs
# (dev: run_probe_eval_cn_vi_full.sh / _sanity.sh — stale path, fix before reuse)
```

### 4. AUROC (CPU, probe venv) — transcript-TPR / turn-FPR, ambiguous excluded
`--split` is `dev`, `test`, or `pool` (pool = test Bloom folders + 3 OOD benchmarks).
The probe script prints layers 27/29, full + reasoning-only (reasoning-only is derived
post-hoc from the same token scores — no extra GPU run).

```bash
# probe venv active
python bloom/compute_haiku_auroc.py --split test     # EN/CN/VI × {nodef,indtax} × {full,ronly}
python bloom/compute_haiku_auroc.py --split pool
python bloom/compute_probe_auroc.py --split test     # probe ensemble, layers 27/29, full + ronly
python bloom/compute_probe_auroc.py --split pool
```

### 5. Translation-faithfulness audit (Opus 4.6, probe venv)
Confirms the translation neither injects misalignment (benign side) nor silently drops it
(positive side). The honest metric is **true translation loss** = positives present in the
EN turn but gone in VI (excludes GT turn-localization mismatches).

```bash
# probe venv active
python bloom/audit_faithfulness.py vi --split all   # full-turn VI audit → label_stability_vi_full.json
python bloom/audit_faithfulness.py en               # EN baseline (positives) → label_baseline_en.json
python bloom/audit_faithfulness.py crosstab         # cross-tab → true translation-loss rate (no API)
```

---

## Outputs / results

- **Probe token scores**: `probe_eval/results/v4_v2_6_combined_v2_span/.../token_scores.json`
- **Judge scores**: `rollout_misalignment_turns_haiku_history_*_score.json` next to each `rollout.json`
- **AUROC tables**: printed by the `compute_*` scripts (also the numbers in the paper table `tab:cross-lingual-auroc`)
- **Audit verdicts**: `bloom/label_stability_vi_full.json`, `bloom/label_baseline_en.json`

Final paper numbers (EN→VI): probe degrades −0.04…−0.09 vs the judge's −0.06…−0.19;
pooled test probe VI AUROC (0.890 full / 0.887 reasoning-only) beats the judge in every
variant. Faithfulness: benign preserved 99.5%; true translation loss 12.7% (dev) / 2.8% (test).

## Notes / gotchas
- The four unified scripts above (`translate_reasoning.py`, `compute_haiku_auroc.py`,
  `compute_probe_auroc.py`, `audit_faithfulness.py`) use the current repo path. A few older
  single-purpose analysis scripts left in `bloom/` (`compute_auroc_cn_vs_en.py`,
  `compute_auroc_ronly.py`, `check_label_stability*.py`, `retranslate_failed_with_opus.py`)
  predate this consolidation and may hardcode the former `persona_vectors` ROOT — fix before reuse.
- For SLURM, keep the probe venv on shared/networked storage, not a node-local home dir,
  or a job on another node fails with `python: No such file or directory`.
- When installing with `uv` onto networked (NFS) storage, prefer `UV_LINK_MODE=hardlink`
  (co-locate the uv cache with the venv); `copy` mode intermittently leaves partial installs.
