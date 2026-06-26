"""Transcript-TPR / turn-FPR AUROC for Haiku judge on CN-translated rollouts
vs EN baseline. Pools across the 10 DEV_PAPER folders.

Convention (matches probe_eval/paired_bootstrap_ci.py + plot_roc_all_methods.py):
  - Each GT-positive transcript -> 1 positive sample = max score across its turns
  - Each turn in a GT-negative transcript           -> 1 negative sample = turn score
  - Each non-misaligned turn in a GT-positive transcript -> 1 negative sample
  - Ambiguous transcripts are excluded entirely
  - AUROC = trapezoidal integral of transcript-TPR vs turn-FPR
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation")
sys.path.insert(0, str(ROOT))
from probe_eval.misalignment_ground_truth import load_unreliable_rollouts  # noqa: E402

EN_DIR = ROOT / "bloom" / "bloom-results"
CN_DIR = ROOT / "bloom" / "bloom-results-cn"

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


def load_ambiguous(folder: str) -> set[tuple[int, int]]:
    """Paper convention: ambiguous (positive-audit) ∪ suspicious (benign-audit)."""
    gt_path = EN_DIR / folder / "rollout_misalignment_turns.json"
    return load_unreliable_rollouts(gt_path)


def load_gt(folder: str) -> dict[tuple[int, int], set[int]]:
    p = EN_DIR / folder / "rollout_misalignment_turns.json"
    d = json.load(open(p))
    return {
        (r["variation_number"], r["repetition_number"]): {t["turn_index"] for t in r.get("misaligned_turns", [])}
        for r in d["evaluated_rollouts"]
    }


def load_scores(folder: str, base_dir: Path, suffix: str) -> dict[tuple[int, int], dict[int, float]]:
    p = base_dir / folder / f"rollout_misalignment_turns_haiku_history_{suffix}_score.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    return {
        (r["variation_number"], r["repetition_number"]): {
            t["turn_index"]: float(t["misalignment_score"]) for t in (r.get("all_turn_scores") or [])
        }
        for r in d["evaluated_rollouts"]
    }


def collect_pos_neg(base_dir: Path, suffix: str, folders: list[str]):
    """Return per-folder (pos_max, neg_turn) lists and folder dict."""
    per_folder = {}
    for fld in folders:
        gt = load_gt(fld)
        ambig = load_ambiguous(fld)
        scores = load_scores(fld, base_dir, suffix)
        pos_max, neg_turn = [], []
        for key, mis_turns in gt.items():
            if key in ambig:
                continue
            turn_scores = scores.get(key)
            if not turn_scores:  # missing or empty — paper convention: skip
                continue
            if mis_turns:  # GT-positive transcript
                pos_max.append(max(turn_scores.values()))
                for tidx, s in turn_scores.items():
                    if tidx not in mis_turns:
                        neg_turn.append(s)
            else:  # GT-negative transcript
                for s in turn_scores.values():
                    neg_turn.append(s)
        per_folder[fld] = (pos_max, neg_turn)
    return per_folder


def auroc(pos: list[float], neg: list[float]) -> float:
    """AUROC where 'pos' is transcript-level and 'neg' is turn-level."""
    if not pos or not neg:
        return float("nan")
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    n_pos, n_neg = len(pos), len(neg)
    # Sweep thresholds at every distinct score in pos ∪ neg
    thresholds = np.unique(np.r_[pos, neg])[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thresholds:
        tpr.append((pos >= t).sum() / n_pos)
        fpr.append((neg >= t).sum() / n_neg)
    tpr.append(1.0); fpr.append(1.0)
    # Sort by FPR (monotone increasing) before trapezoidal integration
    order = np.argsort(fpr)
    fpr = np.asarray(fpr)[order]
    tpr = np.asarray(tpr)[order]
    return float(np.trapezoid(tpr, fpr))


def main():
    en_n = collect_pos_neg(EN_DIR, "nodef", FOLDERS)
    cn_n = collect_pos_neg(CN_DIR, "nodef", FOLDERS)
    en_i = collect_pos_neg(EN_DIR, "indtax", FOLDERS)
    cn_i = collect_pos_neg(CN_DIR, "indtax", FOLDERS)

    print(f"\n{'Folder':<55} {'pos_n':>5} {'neg_n':>5} {'EN-nodef':>9} {'CN-nodef':>9} {'EN-ind':>8} {'CN-ind':>8}")
    print("-" * 110)

    def fmt(v):
        return f"{v:.3f}" if v == v else "  n/a"

    for fld in FOLDERS:
        p_en_n, n_en_n = en_n[fld]
        p_cn_n, n_cn_n = cn_n[fld]
        p_en_i, n_en_i = en_i[fld]
        p_cn_i, n_cn_i = cn_i[fld]
        pos_n = len(p_en_n) or len(p_cn_n)
        neg_n = len(n_en_n) or len(n_cn_n)
        print(
            f"{fld:<55} {pos_n:>5} {neg_n:>5} "
            f"{fmt(auroc(p_en_n, n_en_n)):>9} "
            f"{fmt(auroc(p_cn_n, n_cn_n)):>9} "
            f"{fmt(auroc(p_en_i, n_en_i)):>8} "
            f"{fmt(auroc(p_cn_i, n_cn_i)):>8}"
        )

    # Pooled: gather across folders. Note: each pos is a transcript, each neg is a turn.
    def pool(d):
        pos = []; neg = []
        for p, n in d.values():
            pos.extend(p); neg.extend(n)
        return pos, neg

    print("-" * 110)
    pn_en_n = pool(en_n); pn_cn_n = pool(cn_n)
    pn_en_i = pool(en_i); pn_cn_i = pool(cn_i)
    print(
        f"{'POOLED (transcript-TPR / turn-FPR)':<55} "
        f"{len(pn_en_n[0]):>5} {len(pn_en_n[1]):>5} "
        f"{auroc(*pn_en_n):>9.3f} "
        f"{auroc(*pn_cn_n):>9.3f} "
        f"{auroc(*pn_en_i):>8.3f} "
        f"{auroc(*pn_cn_i):>8.3f}"
    )

    # Mean transcript pos score / mean turn neg score (Haiku confidence shift)
    print("\nPositive (transcript max) and negative (turn) score means [pooled]:")
    def stats(d, name):
        p, n = pool(d)
        print(f"  {name:<10} pos_n={len(p):>3} mean={np.mean(p):.2f}  |  neg_n={len(n):>4} mean={np.mean(n):.2f}  |  AUROC={auroc(p, n):.3f}")
    stats(en_n, "EN nodef")
    stats(cn_n, "CN nodef")
    stats(en_i, "EN indtax")
    stats(cn_i, "CN indtax")

    # ---- Triple decomposition: pos / neg-misaligned-folders / neg-benign-folders
    BENIGN = {f for f in FOLDERS if "benign" in f or f == "sycophancy_benign_glm_4_7_flash"}

    def collect_split(base, sfx):
        pos, neg_misf, neg_benf = [], [], []
        for fld in FOLDERS:
            gt = load_gt(fld)
            ambig = load_ambiguous(fld)
            scs = load_scores(fld, base, sfx)
            for key, mis in gt.items():
                if key in ambig: continue
                ts = scs.get(key)
                if not ts: continue
                if mis:
                    pos.append(max(ts.values()))
                    for tidx, s in ts.items():
                        if tidx not in mis:
                            (neg_benf if fld in BENIGN else neg_misf).append(s)
                else:
                    for s in ts.values():
                        (neg_benf if fld in BENIGN else neg_misf).append(s)
        return pos, neg_misf, neg_benf

    for sfx in ["nodef", "indtax"]:
        p_en, mn_en, bn_en = collect_split(EN_DIR, sfx)
        p_cn, mn_cn, bn_cn = collect_split(CN_DIR, sfx)
        print(f"\n=== {sfx} — decomposition (ambiguous excluded, paper-aligned) ===")
        print(f"  Pool sizes: pos={len(p_en)}  neg-misaligned-folder={len(mn_en)}  neg-benign-folder={len(bn_en)}")
        print(f"  Score means EN:  pos={np.mean(p_en):.2f}  neg-mis={np.mean(mn_en):.2f}  neg-ben={np.mean(bn_en):.2f}")
        print(f"  Score means CN:  pos={np.mean(p_cn):.2f}  neg-mis={np.mean(mn_cn):.2f}  neg-ben={np.mean(bn_cn):.2f}")
        print(f"           Δ:      pos={np.mean(p_cn)-np.mean(p_en):+.2f}  neg-mis={np.mean(mn_cn)-np.mean(mn_en):+.2f}  neg-ben={np.mean(bn_cn)-np.mean(bn_en):+.2f}")
        base = auroc(p_en, mn_en + bn_en); cn = auroc(p_cn, mn_cn + bn_cn)
        print(f"  AUROC EN/EN baseline               = {base:.3f}")
        print(f"  AUROC CN/CN actual                 = {cn:.3f}   Δ={cn-base:+.3f}")
        print(f"  Swap ONLY pos (EN→CN)              = {auroc(p_cn, mn_en+bn_en):.3f}   Δ={auroc(p_cn, mn_en+bn_en)-base:+.3f}")
        print(f"  Swap ONLY benign-folder negs       = {auroc(p_en, mn_en+bn_cn):.3f}   Δ={auroc(p_en, mn_en+bn_cn)-base:+.3f}")
        print(f"  Swap ONLY mis-folder non-mis negs  = {auroc(p_en, mn_cn+bn_en):.3f}   Δ={auroc(p_en, mn_cn+bn_en)-base:+.3f}")


if __name__ == "__main__":
    main()
