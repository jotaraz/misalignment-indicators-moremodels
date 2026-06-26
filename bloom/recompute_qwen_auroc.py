"""Pure-CPU recompute of Qwen dev / test / all-9 AUROC from cached per-turn score
JSONs (no model, no GPU). Used to compare the last-step vs max-step turn reduction
with byte-identical pooling logic, so any AUROC delta is attributable only to the
scoring change.

  python recompute_qwen_auroc.py --dev <dev.json> --test <test.json> [--ood <ood.json>]
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
sys.path.insert(0, str(ROOT / "bloom"))
sys.path.insert(0, str(ROOT))
from score_ood_and_merge_qwen import pool_from_scores, OOD, INDIST  # noqa: E402
from score_auroc_qwen_perturn import auroc  # noqa: E402

B = ROOT / "bloom"


def folder_scores(jpath, folder):
    d = json.load(open(jpath))
    if folder not in d:
        return None
    return {tuple(int(x) for x in k.split("_")): v for k, v in d[folder].items()}


def pools_indist(jpath, base, prefix):
    pools = defaultdict(lambda: ([], []))
    for bch in INDIST:
        for suff in ["", "_benign"]:
            folder = f"{prefix}{bch}{suff}_qwen3_6_35b_a3b"
            sc = folder_scores(jpath, folder)
            gtf = B / base / folder / "rollout_misalignment_turns.json"
            if sc is None or not gtf.exists():
                print(f"  (skip {folder}: scores={'ok' if sc else 'MISSING'} gt={gtf.exists()})")
                continue
            pool_from_scores(sc, gtf, pools)
    return pools


def report(label, pools):
    print(f"\n=== {label} ===  {'view':<8}{'pos':>6}{'neg':>7}{'L23':>8}{'L25':>8}")
    for view in ("full", "ronly", "oonly"):
        p23, n23 = pools[(23, view)]
        p25, n25 = pools[(25, view)]
        print(f"{'':>10}{view:<8}{len(p23):>6}{len(n23):>7}{auroc(p23, n23):>8.3f}{auroc(p25, n25):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev")
    ap.add_argument("--test")
    ap.add_argument("--ood", default=str(B / "qwen36_ood_scores.json"))
    a = ap.parse_args()

    if a.dev:
        report("DEV (in-distribution 5)", pools_indist(a.dev, "bloom-results", ""))

    if a.test:
        test_pools = pools_indist(a.test, "bloom-results-test", "test_")
        report("TEST (in-distribution 5)", test_pools)
        # all-9 = test 5 + 4 OOD (OOD scores are single-turn, unaffected by step_reduce)
        ood = json.load(open(a.ood))
        allp = defaultdict(lambda: ([], []))
        for k, (p, n) in test_pools.items():
            allp[k][0].extend(p)
            allp[k][1].extend(n)
        for name, d in OOD.items():
            if name not in ood:
                print(f"  (ood {name} not cached)")
                continue
            sc = {tuple(int(x) for x in k.split("_")): v for k, v in ood[name].items()}
            pool_from_scores(sc, d / "rollout_misalignment_turns.json", allp)
        report("ALL-9 (test 5 + 4 OOD)", allp)


if __name__ == "__main__":
    main()
