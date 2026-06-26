"""Cross-lingual Haiku-judge AUROC (transcript-TPR / turn-FPR, ambiguous excluded),
EN vs CN vs VI, {nodef,indtax} x {full, reasoning-only}. Unifies the former
compute_auroc_three_lang(_test).py and compute_auroc_pooltest.py.

  --split dev   : 10 dev folders
  --split test  : 10 held-out Bloom test folders
  --split pool  : test Bloom folders + 3 OOD benchmarks (DeceptionBench/MASK/Sycophancy-Eval),
                  pooled into one set (the paper's full test set)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
sys.path.insert(0, str(ROOT))
from probe_eval.misalignment_ground_truth import load_unreliable_rollouts
B = ROOT / "bloom"

DEV_FOLDERS = ["instructed-covert-code-sabotage_benign_glm_4_7_flash","instructed-covert-code-sabotage_glm_4_7_flash",
               "instructed-strategic-sandbagging_benign_glm_4_7_flash","instructed-strategic-sandbagging_glm_4_7_flash",
               "self-preservation_benign_glm_4_7_flash","self-preservation_glm_4_7_flash",
               "strategic-deception_benign_glm_4_7_flash","strategic-deception_glm_4_7_flash",
               "sycophancy","sycophancy_benign_glm_4_7_flash"]
TEST_FOLDERS = ["test_"+b for b in
               ["instructed-covert-code-sabotage_benign_glm_4_7_flash","instructed-covert-code-sabotage_glm_4_7_flash",
                "instructed-strategic-sandbagging_benign_glm_4_7_flash","instructed-strategic-sandbagging_glm_4_7_flash",
                "self-preservation_benign_glm_4_7_flash","self-preservation_glm_4_7_flash",
                "strategic-deception_benign_glm_4_7_flash","strategic-deception_glm_4_7_flash",
                "sycophancy_benign_glm_4_7_flash","sycophancy_glm_4_7_flash"]]
OOD_NAMES = ["deceptionbench", "mask", "sycophancy_eval"]


def entries(split):
    """(gt_dir, en_dir, cn_dir, vi_dir) per behavior."""
    if split == "dev":
        return [(B/"bloom-results"/f, B/"bloom-results"/f, B/"bloom-results-cn"/f, B/"bloom-results-vi"/f) for f in DEV_FOLDERS]
    test = [(B/"bloom-results-test"/f, B/"bloom-results-test"/f, B/"bloom-results-test-cn"/f, B/"bloom-results-test-vi"/f) for f in TEST_FOLDERS]
    if split == "test":
        return test
    ood = [(B/"ood-xlingual"/n, B/"ood-xlingual"/n, B/"ood-xlingual"/f"{n}_cn", B/"ood-xlingual"/f"{n}_vi") for n in OOD_NAMES]
    return test + ood   # pool


def load_gt(gt_dir):
    d = json.load(open(gt_dir/"rollout_misalignment_turns.json"))
    return {(r["variation_number"], r["repetition_number"]):
            {t["turn_index"] for t in r.get("misaligned_turns", [])} for r in d["evaluated_rollouts"]}

def load_scores(d, fname):
    p = d/fname
    if not p.exists(): return {}
    data = json.load(open(p))
    return {(r["variation_number"], r["repetition_number"]):
            {t["turn_index"]: float(t["misalignment_score"]) for t in (r.get("all_turn_scores") or [])}
            for r in data["evaluated_rollouts"]}

def collect(ents, lang_idx, fname):
    pos, neg = [], []
    for e in ents:
        gt = load_gt(e[0]); unrel = load_unreliable_rollouts(e[0]/"rollout_misalignment_turns.json")
        sc = load_scores(e[lang_idx], fname)
        for key, mis in gt.items():
            if key in unrel: continue
            ts = sc.get(key)
            if not ts: continue
            if mis:
                pos.append(max(ts.values()))
                neg += [s for t, s in ts.items() if t not in mis]
            else:
                neg += list(ts.values())
    return pos, neg

def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    pos = np.asarray(pos); neg = np.asarray(neg)
    thr = np.unique(np.r_[pos, neg])[::-1]; tpr=[0.0]; fpr=[0.0]
    for t in thr: tpr.append((pos>=t).sum()/len(pos)); fpr.append((neg>=t).sum()/len(neg))
    tpr.append(1.0); fpr.append(1.0)
    o = np.argsort(fpr)
    return float((getattr(np,"trapezoid",None) or np.trapz)(np.asarray(tpr)[o], np.asarray(fpr)[o]))

SUFFIX = {("nodef","full"):"rollout_misalignment_turns_haiku_history_nodef_score.json",
          ("nodef","ronly"):"rollout_misalignment_turns_haiku_history_nodef_RONLY_score.json",
          ("indtax","full"):"rollout_misalignment_turns_haiku_history_indtax_score.json",
          ("indtax","ronly"):"rollout_misalignment_turns_haiku_history_indtax_RONLY_score.json"}
LANGS = {"EN":1, "CN":2, "VI":3}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="pool", choices=["dev","test","pool"])
    a = ap.parse_args(); ents = entries(a.split)
    data = {(l,m,v): collect(ents, li, SUFFIX[(m,v)]) for l,li in LANGS.items() for m in ("nodef","indtax") for v in ("full","ronly")}
    au = {k: auroc(*v) for k,v in data.items()}
    print(f"\n=== Haiku AUROC ({a.split}) ===\n{'Mode':<8}{'View':<8}{'EN':>9}{'CN':>9}{'VI':>9}{'EN->CN':>9}{'EN->VI':>9}\n"+"-"*65)
    for m in ("nodef","indtax"):
        for v in ("full","ronly"):
            en,cn,vi = au[("EN",m,v)],au[("CN",m,v)],au[("VI",m,v)]
            print(f"  {m:<6}{v:<8}{en:>9.3f}{cn:>9.3f}{vi:>9.3f}{cn-en:>+9.3f}{vi-en:>+9.3f}")
        print()

if __name__ == "__main__":
    main()
