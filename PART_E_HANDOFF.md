# Part E handoff — cross-model probe evaluation

Status as of 2026-06-26. This repo (`jotaraz/misalignment-indicators-moremodels`, forked
from `safety-research/misalignment-indicators`) extends the v2.6 misalignment-indicator
probe pipeline to **4 new target models**, run on the **MPI HTCondor cluster**.

Cluster repo: `/fast/jtaraz/LIARS/misalignment-indicators-moremodels`
Envs: `/fast/jtaraz/envs/{probe,bloom}` · HF cache: `/fast/jtaraz/huggingface` · `.env`: `bloom/.env`

---

## The 4 target models (the registry is the source of truth)

`bloom/cross_model_registry.py` maps each target across its 4 names. Import `MODELS`,
`CONV_BEHAVIORS`, `OOD_BENCHMARKS`, and the `dev_rollout_dirs()` / `ood_rollout_dirs()` helpers.

| key | probe ModelName | probe dir | layers | OpenRouter key | rollout slug | reasoning |
|---|---|---|---|---|---|---|
| llama-70b-r1 | llama-70b-r1 | probes_llama_70b_r1 | 48,50 | deepseek-r1-llama-70b | deepseek_r1_llama_70b | yes |
| llama-70b-3.3 | llama-70b-3.3 | probes_llama_70b_3_3 | 48,50 | llama-3.3-70b (paid) | llama_3_3_70b | no |
| gemma-27b | gemma-27b | probes_gemma_27b | 27,29 | gemma-2-27b | gemma_2_27b | no |
| mistral-24b | mistral-24b | probes_mistral_24b | 23,25 | mistral-small-24b | mistral_small_24b | no |

## SCOPE DECISION (critical — see memory `cross-model-agentic-scope`)

The new OpenRouter models are evaluated **only on tool-free behaviors**:
- **Dev/test (2)**: `sycophancy`, `instructed-strategic-sandbagging`
- **OOD (4)**: deception-bench, mask, sycophancy-eval, agentic-misalignment (single-turn replays)

**Excluded** (tool-using → fail for these models; stay GLM/Qwen-only): `self-preservation`
(simenv), `instructed-covert-code-sabotage` (simenv), `strategic-deception` (function-calling).
Reason: bloom's simenv requires function-calling. The `core.py` `FUNCTION_CALLING_WHITELIST`
gate was fixed (these models are whitelisted now), but the *actual* tool rollout then hangs
(free endpoints) or dies silently (paid) → 0 target turns. GLM/Qwen work; these don't.

R1 reasoning IS captured correctly in conversation modality (5/5 turns, same `{type:reasoning}`
block format as GLM) → **use the standard `probe_eval.evaluate` scorer for all 4** (no Qwen-style
per-turn scorer needed). This is the big simplification vs the original Qwen §8 path.

---

## Done

- **A** Model registration — `…/deception_detection/models.py` (6 models; loaders fixed for cluster download; R1 uses its own tokenizer).
- **B** Training — pipeline validated (gemma-27b smoke probe, 0.98 val AUROC). `scripts/cluster/{run_train.sh,train.sub,submit_all_train.sh}`. **But the full fan-out exposed per-model template/OOM issues — see next section.**
- **C** Rollout + GT — validated; **dev rollouts succeeded for all 4 models** (sycophancy + sandbagging, misaligned+benign). `bloom/cross_model_rollout.py` (+ `--limit`, `--split` TBD), `bloom/cross_model_gt.sh`, `scripts/cluster/{run_rollout_gt.sh,rollout_gt.sub,submit_all_rollout_gt.sh}`. Restricted to the 2 conv behaviors.
- **E foundation** — `bloom/cross_model_registry.py`; dev scoring `scripts/cluster/{run_score.sh,submit_all_score.sh}` (uses `evaluate.py`).

## Training fan-out outcome — per-model template status (2026-06-26)

The pipeline is built around GLM/Qwen/Llama-3/Gemma chat templates (`tokenized_data.py`
hardcodes a detect-mask regex per `ModelType`). Older-style templates work; the two newest
models fight it. The first full fan-out (`submit_all_train.sh`) result:

| Model | Training status | Detail |
|---|---|---|
| **gemma-27b** | ✅ works (needs 2 GPUs) | CUDA OOM on 1×H100; fixed by `request_gpus = 2`. Re-run `submit_all_train.sh` style with 2 GPUs for gemma. |
| **llama-3.3-70b** | ✅ works | standard Llama-3 template; trains clean. |
| **mistral-24b** | ⚠️ partial fix | Regex extended for Mistral-Small-3 `[SYSTEM_PROMPT]` + `fix_mistral_regex=True` (committed). REMAINING BUG: `TokenizedDataset.from_dialogue_list` formats mistral with a **default system prompt** ("You are Mistral Small 3…") instead of the dialogue's system message → the content-verbatim assertion (`tokenized_data.py:224`) fails. `apply_chat_template` alone preserves the real system content, so the divergence is inside `from_dialogue_list`'s mistral formatting/template_kwargs path — trace there. |
| **llama-70b-r1** | ❌ UNFIXABLE for probe training | DeepSeek-R1 template (a) uses `<｜User｜>`/`<｜Assistant｜>` tokens (wrong regex branch — it's `type=="llama"`), and (b) **strips `<think>` reasoning from assistant history** (`.split('</think>')[-1]` baked into the template). So the synthetic reasoning spans vanish and the model never ingests reasoning → a reasoning probe cannot be trained. No flag disables the stripping. **Recommend dropping R1 entirely** (it also failed agentic). |

**Open decisions (pending):** (1) drop R1 from the whole cross-model set? (recommended);
(2) invest one more debugging pass on the mistral default-system-prompt quirk, or settle for
**gemma + llama-3.3** as the trained set. Either way the trained cross-model probes are
**non-reasoning only** (GLM/Qwen remain the reasoning models).

To resume training after fixes: `bash scripts/cluster/submit_all_train.sh` (give gemma 2 GPUs;
drop R1 from the submit list; include mistral only once the from_dialogue_list fix lands).
`train.py` skips already-trained indicators, so re-runs resume.

## Remaining Part E (build + VALIDATE against landed data — don't build blind)

The smoke test caught 9 bugs precisely by validating against live data. Do the same here.

1. **Dev scoring** (foundation built, untested): once probes + dev rollouts land, run
   `bash scripts/cluster/submit_all_score.sh`. Verify `probe_eval.evaluate` parses the
   cross-model rollout dirs and writes per-token scores under `probe_eval/results_<model>/`.
   First-run gotchas to check: output path keyed on `Path(rollout_dir).name`; `evaluate.py`
   reads the model from each probe `cfg.yaml`.

2. **OOD replay** — generalize `ood_misalignment_eval/replay_qwen_ood.py` and
   `replay_qwen_sycophancy.py` to take `--model` (look up `bloom_key` + `dir_slug` from the
   registry; remove the hardcoded `MODEL = "qwen/..."` and `ROOT = /workspace-vast/kaiwen/...`).
   They replay a committed GLM OOD `rollout.json`'s target prompts through the new model via
   AsyncOpenAI and capture reasoning as a `<think>` block. Write to `ood_rollout_dirs(model)`.
   For agentic-misalignment/sycophancy-eval/mask/deception-bench use the GLM rollout as the
   prompt source (`*/rollouts/glm-4.7-flash/...`).

3. **OOD ground truth** — re-judge each model's replayed responses (decided: re-judge, not
   reuse benchmark labels). Run `black_box_ind_judge` per OOD dir (mirror `cross_model_gt.sh`).

4. **OOD scoring + AUROC merge** — generalize `bloom/score_ood_and_merge_qwen.py` to `--model`
   (parametrize the `OOD` dict via `ood_rollout_dirs`, `INDIST` via `CONV_BEHAVIORS`, layers +
   probe root via registry; fix the hardcoded `ROOT`). OOD is single-turn so per-turn ==
   whole-dialogue. Report **all-6** AUROC (2 dev + 4 OOD) per model, pooled with the
   transcript-TPR / turn-FPR convention.

5. **Test split** — extend `cross_model_rollout.py` with `--split test`: reuse the **GLM test
   scenarios** (`bloom-results-test/`, on HF — `kzhou35/misalignment-indicators-bloom-rollouts`,
   pull first) as fixed scenarios, output to `bloom-results-test/test_<behavior>[_benign]_<slug>`.
   `bloom/gen_qwen_test.py` is the Qwen template (hardcoded paths). Then GT + score as for dev.

6. **HTCondor wrappers** — OOD replay+GT = CPU+proxy (like `run_rollout_gt.sh`); OOD/dev scoring
   = GPU (70B models need `request_gpus = 2`, like `submit_all_score.sh`).

7. **(optional) Thresholds** — only if you want the discrete-ROC / fixed-FPR operating points
   (AUROC itself is threshold-free). Needs the neutral set `probe/data/neutral/dialogues_filtered_v3.json`,
   which is **gitignored** — rsync it to the cluster (same as the training data), then run
   `probe/neutral/score_neutral.py` + `probe.recompute_thresholds` per model.

---

## Hard-won gotchas (don't re-discover these)

- **`tokenized_data.py` hardcodes a detect-mask regex per chat template** (`_prefix_pattern`,
  `ModelType`). New templates break it: Mistral-Small-3 added `[SYSTEM_PROMPT]` (regex fixed);
  DeepSeek-R1 uses `<｜User｜>` tokens AND strips `<think>` from history (unfixable). The
  `_get_detection_mask` algorithm also requires each message's content to appear **verbatim**
  in the formatted text (`:224`) — any template that rewrites/strips/injects content breaks it
  (e.g. R1 reasoning-strip; mistral default-system-prompt). When adding a model, render its
  template first (`tok.apply_chat_template`) and check both the prefix tokens and verbatim content.
- **R1-distill can't train reasoning probes** — its template removes `<think>` from assistant
  history, so the model never sees the reasoning the probe must read. Not a code bug.

- **`probe/data/` and `probe_eval/results/` are gitignored.** The upstream repo force-added the
  training data; the fresh-init `git add -A` dropped it. Training data + neutral set live ONLY on
  the cluster (rsync'd), not in git. The malicious-action training JSON contains synthetic
  Stripe-key strings → **GitHub push protection blocks committing it** (keep it out of the repo).
- **Never `git add -A`** — it sweeps rsync'd rollout data into commits. Cross-model rollout/probe
  artifacts are gitignored (`bloom/bloom-results/*_<slug>/`, `probe/probes_<slug>/`); add files
  explicitly.
- **Condor clean env**: no `HOME`, minimal `PATH`. The wrappers default both (`: "${HOME:=…}"`,
  prepend coreutils). HF on `/fast` needs `SOFTFILELOCK=1` + `scripts/cluster/sitecustomize.py`.
- **Evaluator model** in rollouts must be `claude-opus-4.6` (GLM metadata sometimes names
  `claude-opus-4.5`, absent from models.json). Rollout `max_tokens >= 16000` (must exceed the
  reasoning thinking budget).
- **llama-3.3 uses the PAID endpoint** (`:free` rate-limit-hangs).
- `requirements-probe.txt` needed: jaxtyping, peft, python-dotenv, termcolor, plotly, goodfire
  (deception_detection import chain).

## Cluster cheatsheet

```bash
ssh jtaraz@login.cluster.is.localnet
cd /fast/jtaraz/LIARS/misalignment-indicators-moremodels && git pull
condor_q jtaraz                       # job status (I=idle R=running H=held)
bash scripts/cluster/submit_all_score.sh   # GPU dev scoring (after data lands)
# logs: logs/{train,eval,score}_<model>.{out,err}
```
