# GLM-4.7-Flash & Qwen3.6-35B-A3B — Probe Ensemble Results

18-indicator probe ensemble (v4_v2_6_combined_v2_span), OR-fused per turn, scored on
the **dev** Bloom set (5 behaviors × {misaligned, benign} = 10 rollout dirs per model).

- **GLM-4.7-Flash** — probes `probe/probes_paper_main`, layers **27 / 29**
- **Qwen3.6-35B-A3B** — probes `probe/probes_qwen36`, layers **23 / 25**

**Metric convention (transcript-TPR / turn-FPR).** Positives = the transcript-level
score (max over turns, each turn = max over sentences over the 18 OR-fused indicators)
on transcripts that contain a ground-truth misaligned turn. Negatives = turn-level
scores on every non-misaligned turn of misaligned transcripts plus all turns of benign
transcripts. Unreliable/ambiguous rollouts excluded (`load_unreliable_rollouts`).
Reasoning-only (`ronly`) re-partitions each turn to its reasoning tokens post-hoc.
TPR@FPR thresholds are set on the negatives to hit each target FPR, then the achieved
transcript-level TPR is reported.

Generated with `bloom/compute_probe_auroc.py --model {glm,qwen} --split dev`
(English rollouts; CN/VI columns N/A).

---

## GLM-4.7-Flash

Positives = 90 misaligned transcripts · Negatives = 737 turns.

| Layer | View | AUROC | TPR @ 1% FPR | TPR @ 5% FPR | TPR @ 10% FPR |
|------:|:-----|:-----:|:------------:|:------------:|:-------------:|
| 27 | full  | **0.969** | 0.667 | 0.856 | 0.922 |
| 27 | ronly | 0.969 | 0.656 | 0.844 | 0.922 |
| 29 | full  | **0.966** | 0.600 | 0.800 | 0.922 |
| 29 | ronly | 0.966 | 0.600 | 0.811 | 0.922 |

---

## Qwen3.6-35B-A3B

Positives = 54 misaligned transcripts · Negatives = 677 turns (full) / 192 (reasoning-only).

| Layer | View | AUROC | TPR @ 1% FPR | TPR @ 5% FPR | TPR @ 10% FPR |
|------:|:-----|:-----:|:------------:|:------------:|:-------------:|
| 23 | full  | **0.953** | 0.648 | 0.796 | 0.852 |
| 23 | ronly | 0.848 | 0.426 | 0.611 | 0.630 |
| 25 | full  | **0.945** | 0.667 | 0.796 | 0.870 |
| 25 | ronly | 0.835 | 0.296 | 0.611 | 0.630 |

---

## Notes

- **Where the signal lives.** For **GLM**, full ≈ reasoning-only (0.969 vs 0.969 AUROC
  at L27) — detectable misalignment is essentially all in the reasoning. For **Qwen**,
  reasoning-only is markedly weaker (0.953 → 0.848 AUROC; TPR@1%FPR 0.65 → 0.43 at L23) —
  much of Qwen's detectable signal is in the output, not the CoT.
- **Coverage.** 720/720 token-score files (18 indicators × 2 layers × 10 dirs × 2 models),
  0 extraction failures. Dialogues are the post-`load_rollout_data` counts (malformed/empty
  transcripts dropped by the standard filter), not raw `transcript_*.json` file counts.
- **Scoring config.** GLM scored on 1×H100 (`cut_at_layer=30`); Qwen on 2×H100
  (`cut_at_layer=26`, `ATTN_IMPL=sdpa` to avoid eager-attention O(seq²) OOM on long
  rollouts). batch_size 1.
