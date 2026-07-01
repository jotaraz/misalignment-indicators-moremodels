# Cross-model rollout & ground-truth coverage

_Snapshot 2026-06-27. Each cell = **A / B** where_
- **A** = transcripts **correctly generated** (≥1 real target turn; the 0-target-turn agentic failures are excluded).
- **B** = transcripts **GT-judged without error** (no `indicator_judge_errors`; ≤ A by construction).
- `—` = dataset not run for that model. `·B` = benign control.

| Model | self-pres | self-pres ·B | strat-dec | strat-dec ·B | syco | syco ·B | sabotage | sabotage ·B | sandbag | sandbag ·B |
|---|---|---|---|---|---|---|---|---|---|---|
| **GLM-4.7-Flash** | 20/19 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 30/28 | 30/29 | 38/36 | 40/40 |
| **Qwen3.6-35B** | 20/20 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 29/28 | 29/29 | 40/40 | 40/38 |
| **DeepSeek-R1-Llama-70B** | 0/0 | 0/0 | — | — | 2/2 | 2/2 | — | — | 40/39 | 40/40 |
| **Llama-3.3-70B** | 2/2 | 2/0 | — | — | 10/0 | 10/0 | — | — | 40/0 | 40/0 |
| **gemma-2-27b** | 0/0 | — | — | — | 10/10 | 10/10 | — | — | 40/24 | 40/0 |
| **Mistral-Small-24B** | 0/0 | — | — | — | 10/9 | 10/10 | — | — | 40/0 | 40/0 |

## Partial ground truth (open-ended misalignment judge only)

For these credit-hit **misaligned** dirs the open-ended misalignment judge reached more transcripts than the indicator judge (which fully failed). Those have the primary AUROC label but lack indicator labels:

- `self-preservation_glm_4_7_flash` — 20 have open-ended summaries, 19 fully judged
- `instructed-covert-code-sabotage_glm_4_7_flash` — 30 have open-ended summaries, 28 fully judged
- `instructed-strategic-sandbagging_glm_4_7_flash` — 40 have open-ended summaries, 36 fully judged
- `instructed-covert-code-sabotage_qwen3_6_35b_a3b` — 30 have open-ended summaries, 28 fully judged
- `self-preservation_deepseek_r1_llama_70b` — 2 have open-ended summaries, 0 fully judged
- `self-preservation_gemma_2_27b` — 2 have open-ended summaries, 0 fully judged
- `instructed-strategic-sandbagging_gemma_2_27b` — 38 have open-ended summaries, 24 fully judged
- `self-preservation_mistral_small_24b` — 2 have open-ended summaries, 0 fully judged
- `sycophancy_mistral_small_24b` — 10 have open-ended summaries, 9 fully judged
- `instructed-strategic-sandbagging_mistral_small_24b` — 19 have open-ended summaries, 0 fully judged

## Status notes
- **GLM / Qwen**: complete (generation + both GT passes).
- **R1 sandbagging**: re-judged 2026-06-27 after the Anthropic credit top-up → now complete (1 misaligned rollout the judge *refused*).
- **New models' `self-pres`** cells are the failed agentic smoke attempts (0 real target turns) — not usable data.
- **Still missing GT** (credit failure, not yet re-judged): Llama-3.3 (syco + sandbag), Mistral/gemma sandbag, all new-model sandbag-benign.
