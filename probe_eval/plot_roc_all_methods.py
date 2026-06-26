"""ROC comparison across all 6 methods on the 15-folder paper test set.

Methods:
  - LLM Judge (Qwen3 / Qwen3-ind / Haiku / Haiku-ind) — continuous score sweep
  - Indicator Probes (stage 1) — discrete points at 10 FPR targets
  - Probes + Haiku 4.5 cascade (stage 2) — discrete points at 10 FPR targets

Plot shows full ROC and a zoomed low-FPR view, pooled across the 15 paper test
behaviors (5 in-distribution × {mis, benign} + 5 OOD evals).
"""
from __future__ import annotations

import io
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from probe_eval.paired_bootstrap_ci import (  # noqa: E402
    CASCADE_FPRS, build_records_continuous, build_records_discrete,
    discover_behaviors_to_rollout_dirs, load_cascade_per_turn_rows,
    load_judge_scores, TEST_PAPER,
)


def operating_points_discrete(recs):
    n_pos = sum(1 for r in recs.values() if r["gt_pos"])
    n_neg = sum(r["neg_total"] for r in recs.values())
    pf = defaultdict(int); nf = defaultdict(int)
    for r in recs.values():
        if r["gt_pos"]:
            for x, f in r["pos_flagged"].items():
                if f: pf[x] += 1
        for x, c in r["neg_flagged"].items(): nf[x] += c
    return [(nf[x] / n_neg, pf[x] / n_pos) for x in CASCADE_FPRS]


def operating_points_continuous(recs):
    """Sweep all distinct turn scores; one (FPR, TPR) per threshold."""
    pos_max = []
    neg_turn = []
    for r in recs.values():
        if r["gt_pos"] and r["pos_max_score"] is not None:
            pos_max.append(r["pos_max_score"])
        neg_turn.extend(r["neg_turn_scores"])
    if not pos_max or not neg_turn:
        return []
    pos = np.sort(np.asarray(pos_max, dtype=float))[::-1]
    neg = np.sort(np.asarray(neg_turn, dtype=float))[::-1]
    n_pos, n_neg = len(pos), len(neg)
    thresholds = sorted(set(pos.tolist() + neg.tolist()), reverse=True)
    pts = [(0.0, 0.0)]
    i_p = i_n = 0
    for t in thresholds:
        while i_p < n_pos and pos[i_p] >= t:
            i_p += 1
        while i_n < n_neg and neg[i_n] >= t:
            i_n += 1
        pts.append((i_n / n_neg, i_p / n_pos))
    return pts


def trap_auc(pts):
    p = sorted(set(pts))
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(p, p[1:]))


def main():
    beh_to_rollout = discover_behaviors_to_rollout_dirs()
    cascade_pt = load_cascade_per_turn_rows()
    behaviors = TEST_PAPER

    with contextlib.redirect_stdout(io.StringIO()):
        h_recs = build_records_continuous(
            load_judge_scores(behaviors, "haiku", beh_to_rollout),
            behaviors, beh_to_rollout)
        hi_recs = build_records_continuous(
            load_judge_scores(behaviors, "haiku_ind", beh_to_rollout),
            behaviors, beh_to_rollout)
        q_recs = build_records_continuous(
            load_judge_scores(behaviors, "qwen", beh_to_rollout),
            behaviors, beh_to_rollout)
        qi_recs = build_records_continuous(
            load_judge_scores(behaviors, "qwen_ind", beh_to_rollout),
            behaviors, beh_to_rollout)
    p_recs = build_records_discrete("probe", cascade_pt, set(behaviors))
    c_recs = build_records_discrete("cascade", cascade_pt, set(behaviors))

    methods = [
        ("LLM Judge (Qwen3)",      operating_points_continuous(q_recs),  "#9467bd", "-",  "o"),
        ("LLM Judge (Qwen3, ind)", operating_points_continuous(qi_recs), "#9467bd", "--", "s"),
        ("LLM Judge (Haiku)",      operating_points_continuous(h_recs),  "#ff7f0e", "-",  "o"),
        ("LLM Judge (Haiku, ind)", operating_points_continuous(hi_recs), "#ff7f0e", "--", "s"),
        ("Indicator Probes",       operating_points_discrete(p_recs),    "#1f77b4", "-",  "o"),
        ("Probes + Haiku 4.5",     operating_points_discrete(c_recs),    "#d62728", "-",  "o"),
    ]

    # Compute (anchored) AUROCs
    aucs = {}
    for name, pts, *_ in methods:
        full = sorted(set([(0.0, 0.0)] + pts + [(1.0, 1.0)]))
        aucs[name] = trap_auc(full)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Full ROC -------------------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    for name, pts, color, ls, mk in methods:
        full = sorted(set([(0, 0)] + pts + [(1, 1)]))
        x, y = zip(*full)
        ax.plot(x, y, linestyle=ls, marker=mk, color=color, linewidth=1.8,
                markersize=3, alpha=0.9,
                label=f"{name} (AUC={aucs[name]:.3f})")
    casc_max_fpr = max(x for x, _ in methods[5][1])
    ax.axvline(casc_max_fpr, color="#d62728", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(casc_max_fpr + 0.01, 0.02, f"Cascade ceiling\n(FPR={casc_max_fpr:.3f})",
            fontsize=8, color="#d62728")
    ax.set_xlabel("Turn-level FPR")
    ax.set_ylabel("Transcript-level TPR")
    ax.set_title("Full ROC (0–1)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    # --- Zoom: FPR ≤ 0.2 ------------------------------------------------
    ax = axes[1]
    ax.plot([0, 0.2], [0, 0.2], color="gray", linestyle=":", linewidth=1)
    for name, pts, color, ls, mk in methods:
        full = sorted(set([(0, 0)] + pts + [(1, 1)]))
        x, y = zip(*full)
        ax.plot(x, y, linestyle=ls, marker=mk, color=color, linewidth=1.8,
                markersize=4, alpha=0.9, label=name)
    ax.axvline(casc_max_fpr, color="#d62728", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Turn-level FPR")
    ax.set_ylabel("Transcript-level TPR")
    ax.set_title("Zoom: FPR ∈ [0, 0.2]")
    ax.set_xlim(-0.005, 0.2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        f"Misalignment monitoring ROC — pooled across {len(behaviors)} paper test behaviors",
        fontsize=12,
    )
    fig.tight_layout()
    out = (Path(__file__).resolve().parent / "results"
           / "v4_v2_6_combined_v2_span"
           / f"roc_all_methods_test{len(behaviors)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")
    print()
    print("Method AUROCs (anchored at (0,0) and (1,1)):")
    for name in aucs:
        print(f"  {name:<28} {aucs[name]:.4f}")


if __name__ == "__main__":
    main()
