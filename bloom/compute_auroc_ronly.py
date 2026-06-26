"""Transcript-TPR / turn-FPR AUROC for the 4 conditions:
  EN full,  EN reasoning-only,  CN full,  CN reasoning-only
Each in nodef + indtax.

Tests the hypothesis that the AUROC drop on CN comes from the reasoning↔action
linkage: if reasoning-only narrows the EN-CN gap, action context was carrying
much of the signal in EN that translation breaks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation")
sys.path.insert(0, str(ROOT))
from probe_eval.misalignment_ground_truth import load_unreliable_rollouts  # noqa: E402

EN = ROOT / "bloom" / "bloom-results"
CN = ROOT / "bloom" / "bloom-results-cn"

FOLDERS = [
    "instructed-covert-code-sabotage_benign_glm_4_7_flash",
    "instructed-covert-code-sabotage_glm_4_7_flash",
    "instructed-strategic-sandbagging_benign_glm_4_7_flash",
    "instructed-strategic-sandbagging_glm_4_7_flash",
    "self-preservation_benign_glm_4_7_flash",
    "self-preservation_glm_4_7_flash",
    "strategic-deception_benign_glm_4_7_flash",
    "strategic-deception_glm_4_7_flash",
    "sycophancy",
    "sycophancy_benign_glm_4_7_flash",
]


def load_gt(folder):
    p = EN / folder / "rollout_misalignment_turns.json"
    d = json.load(open(p))
    return {(r["variation_number"], r["repetition_number"]):
            {t["turn_index"] for t in r.get("misaligned_turns", [])}
            for r in d["evaluated_rollouts"]}


def load_scores(folder, base, fname):
    p = base / folder / fname
    if not p.exists():
        return {}
    d = json.load(open(p))
    return {(r["variation_number"], r["repetition_number"]):
            {t["turn_index"]: float(t["misalignment_score"]) for t in (r.get("all_turn_scores") or [])}
            for r in d["evaluated_rollouts"]}


def collect(base, fname):
    pos, neg = [], []
    for fld in FOLDERS:
        gt = load_gt(fld)
        unrel = load_unreliable_rollouts(EN / fld / "rollout_misalignment_turns.json")
        sc = load_scores(fld, base, fname)
        for key, mis in gt.items():
            if key in unrel: continue
            ts = sc.get(key)
            if not ts: continue
            if mis:
                pos.append(max(ts.values()))
                for tidx, s in ts.items():
                    if tidx not in mis:
                        neg.append(s)
            else:
                for s in ts.values():
                    neg.append(s)
    return pos, neg


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    pos = np.asarray(pos); neg = np.asarray(neg)
    thr = np.unique(np.r_[pos, neg])[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append((pos >= t).sum() / len(pos))
        fpr.append((neg >= t).sum() / len(neg))
    tpr.append(1.0); fpr.append(1.0)
    order = np.argsort(fpr); fpr = np.asarray(fpr)[order]; tpr = np.asarray(tpr)[order]
    return float(np.trapezoid(tpr, fpr))


def main():
    runs = {
        ("EN", "nodef", "full"):  (EN, "rollout_misalignment_turns_haiku_history_nodef_score.json"),
        ("EN", "nodef", "ronly"): (EN, "rollout_misalignment_turns_haiku_history_nodef_RONLY_score.json"),
        ("EN", "indtax","full"):  (EN, "rollout_misalignment_turns_haiku_history_indtax_score.json"),
        ("EN", "indtax","ronly"): (EN, "rollout_misalignment_turns_haiku_history_indtax_RONLY_score.json"),
        ("CN", "nodef", "full"):  (CN, "rollout_misalignment_turns_haiku_history_nodef_score.json"),
        ("CN", "nodef", "ronly"): (CN, "rollout_misalignment_turns_haiku_history_nodef_RONLY_score.json"),
        ("CN", "indtax","full"):  (CN, "rollout_misalignment_turns_haiku_history_indtax_score.json"),
        ("CN", "indtax","ronly"): (CN, "rollout_misalignment_turns_haiku_history_indtax_RONLY_score.json"),
    }
    data = {k: collect(*v) for k, v in runs.items()}
    aurocs = {k: auroc(*data[k]) for k in runs}

    print(f"\n{'Mode':<10}{'View':<10}{'EN AUROC':>10}{'CN AUROC':>10}{'Δ (CN−EN)':>13}")
    print("-" * 53)
    for mode in ("nodef", "indtax"):
        for view in ("full", "ronly"):
            en = aurocs[("EN", mode, view)]
            cn = aurocs[("CN", mode, view)]
            print(f"  {mode:<8}{view:<10}{en:>10.3f}{cn:>10.3f}{cn-en:>+13.3f}")
        print()
    # Compare full vs ronly within each language
    print(f"\n{'Mode':<10}{'Lang':<10}{'Full AUROC':>12}{'RONLY AUROC':>13}{'Δ (RONLY−full)':>17}")
    print("-" * 65)
    for mode in ("nodef", "indtax"):
        for lang in ("EN", "CN"):
            f = aurocs[(lang, mode, "full")]
            r = aurocs[(lang, mode, "ronly")]
            print(f"  {mode:<8}{lang:<10}{f:>12.3f}{r:>13.3f}{r-f:>+17.3f}")
        print()

    # Pool sizes
    print("\nPool sizes (paper-aligned, ambiguous excluded):")
    for k in runs:
        p, n = data[k]
        print(f"  {'/'.join(k):<30}  pos_n={len(p):>3}  neg_n={len(n):>4}  "
              f"pos_mean={np.mean(p):.2f}  neg_mean={np.mean(n):.2f}")


if __name__ == "__main__":
    main()
