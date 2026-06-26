"""Score the Qwen OOD benchmark rollouts with the Qwen probe ensemble and report
the all-N test AUROC (5 in-distribution test behaviors + OOD benchmarks).

Reuses the per-turn scorer machinery (score_dir / load_detectors). OOD rollouts are
single-turn benchmark responses, so per-turn == whole-dialogue. In-distribution test
scores are loaded from the cached bloom/qwen36_perturn_scores_test.json; OOD dirs are
scored here on GPU. Pools use the transcript-TPR / per-turn-FPR convention with
unreliable exclusion, exactly as the in-distribution numbers.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
sys.path.insert(0, str(ROOT / "bloom"))
from score_auroc_qwen_perturn import (load_detectors, score_dir, auroc,
                                      get_model_and_tokenizer, ModelName)
sys.path.insert(0, str(ROOT))
from probe_eval.misalignment_ground_truth import load_unreliable_rollouts

B = ROOT / "bloom"
LAYERS = [23, 25]
# OOD benchmark -> rollout dir (each must hold rollout.json + rollout_misalignment_turns.json)
OOD = {
    "deception-bench": ROOT / "ood_misalignment_eval/deception-bench/rollouts/qwen3.6-35b-a3b",
    "mask": ROOT / "ood_misalignment_eval/mask/rollouts/qwen3.6-35b-a3b",
    "sycophancy-eval": ROOT / "ood_misalignment_eval/sycophancy-eval/rollouts/qwen3.6-35b-a3b/sycophancy_feedback_filtered",
    "agentic-misalignment": ROOT / "ood_misalignment_eval/agentic-misalignment/results/ood_eval_qwen/bloom_rollout",
}
INDIST = ["instructed-covert-code-sabotage", "instructed-strategic-sandbagging",
          "self-preservation", "strategic-deception", "sycophancy"]


def pool_from_scores(sc_by_key, gt_path, pools):
    """sc_by_key: {(var,rep,turn): {layer: {view: score}}}; add to pools[(layer,view)]."""
    gt = json.load(open(gt_path))
    unrel = load_unreliable_rollouts(gt_path)
    mis_by = {(r["variation_number"], r["repetition_number"]):
              {t["turn_index"] for t in r.get("misaligned_turns", [])}
              for r in gt["evaluated_rollouts"]}
    def _lk(rec, layer):  # layer keys may be int (in-memory) or str (from JSON)
        return rec.get(layer, rec.get(str(layer), {}))
    by = defaultdict(dict)
    for (v, r, t), rec in sc_by_key.items():
        by[(v, r)][t] = rec
    for layer in LAYERS:
        for view in ("full", "ronly", "oonly"):
            pos, neg = pools[(layer, view)]
            for key, mis in mis_by.items():
                if key in unrel:
                    continue
                ts = by.get(key)
                if not ts:
                    continue
                vals = {t: _lk(rec, layer)[view] for t, rec in ts.items() if view in _lk(rec, layer)}
                if not vals:
                    continue
                if mis:
                    pos.append(max(vals.values())); neg += [s for t, s in vals.items() if t not in mis]
                else:
                    neg += list(vals.values())


def main():
    dets = load_detectors(LAYERS)
    print(f"Loaded {len(dets)} detectors")
    _, tok = get_model_and_tokenizer(ModelName.QWEN_35B, omit_model=True)
    model, _ = get_model_and_tokenizer(ModelName.QWEN_35B)

    indist_pools = defaultdict(lambda: ([], []))
    ood_pools = defaultdict(lambda: ([], []))
    per_dataset = {}

    # in-distribution test: reuse cached scores
    test_scores = json.load(open(B / "qwen36_perturn_scores_test.json"))
    for bch in INDIST:
        for suff in ["", "_benign"]:
            f = f"test_{bch}{suff}_qwen3_6_35b_a3b"
            gtf = B / "bloom-results-test" / f / "rollout_misalignment_turns.json"
            if not gtf.exists() or f not in test_scores:
                continue
            sc = {tuple(int(x) for x in k.split("_")): v for k, v in test_scores[f].items()}
            pool_from_scores(sc, gtf, indist_pools)

    # OOD: score on GPU (cache per-key scores so the merge can be re-run on CPU)
    cache_path = B / "qwen36_ood_scores.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    for name, d in OOD.items():
        gtf = d / "rollout_misalignment_turns.json"
        if not (d / "rollout.json").exists() or not gtf.exists():
            print(f"  (skip {name}: missing rollout.json or GT)"); continue
        if name in cache:
            print(f"Using cached scores for OOD {name}")
            sc = {tuple(int(x) for x in k.split("_")): v for k, v in cache[name].items()}
        else:
            print(f"Scoring OOD {name} ...", flush=True)
            sc = score_dir(d, dets, LAYERS, tok, model)
            cache[name] = {f"{v}_{r}_{t}": val for (v, r, t), val in sc.items()}
            json.dump(cache, open(cache_path, "w"))
        p = defaultdict(lambda: ([], []))
        pool_from_scores(sc, gtf, p)
        pool_from_scores(sc, gtf, ood_pools)
        per_dataset[name] = {f"L{l}_{v}": auroc(*p[(l, v)]) for l in LAYERS for v in ("full", "ronly")}
        per_dataset[name]["n"] = (len(p[(23, "ronly")][0]), len(p[(23, "ronly")][1]))

    # report
    def merged(a, b):
        return {(l, v): (a[(l, v)][0] + b[(l, v)][0], a[(l, v)][1] + b[(l, v)][1])
                for l in LAYERS for v in ("full", "ronly", "oonly")}
    allp = merged(indist_pools, ood_pools)

    print("\n=== Per-OOD-dataset AUROC ===")
    for name, m in per_dataset.items():
        print(f"  {name:<24} n(pos/neg)={m['n']}  ronly L23={m['L23_ronly']:.3f} L25={m['L25_ronly']:.3f}  full L23={m['L23_full']:.3f}")
    for label, pools in [("in-distribution test (5)", indist_pools), ("OOD only", ood_pools), ("ALL", allp)]:
        print(f"\n=== {label} ===  {'view':<8}{'pos':>6}{'neg':>7}{'L23':>8}{'L25':>8}")
        for view in ("full", "ronly", "oonly"):
            p23, n23 = pools[(23, view)]; p25, n25 = pools[(25, view)]
            print(f"{'':>22}{view:<8}{len(p23):>6}{len(n23):>7}{auroc(p23, n23):>8.3f}{auroc(p25, n25):>8.3f}")

    json.dump({k: {kk: (vv if not isinstance(vv, tuple) else list(vv)) for kk, vv in v.items()}
               for k, v in per_dataset.items()}, open(B / "qwen36_ood_auroc.json", "w"), indent=1)
    print(f"\nsaved per-dataset AUROC -> {B/'qwen36_ood_auroc.json'}")


if __name__ == "__main__":
    main()
