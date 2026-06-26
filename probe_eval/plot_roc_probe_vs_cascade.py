"""Plot probe vs cascade ROC curves on test_15, with full and zoomed views.

Shows why standard AUROC compresses the gap: the cascade's measured curve only
covers FPR <= 0.085 (Haiku rejects ~90% of probe fires), so >90% of its AUC
comes from a straight-line interpolation to (1,1), while the probe's data
spans FPR up to 0.81.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from probe_eval.paired_bootstrap_ci import (  # noqa: E402
    CASCADE_FPRS, build_records_discrete, load_cascade_per_turn_rows,
)

TEST_15 = [
    "test_sycophancy_glm_4_7_flash", "test_sycophancy_benign_glm_4_7_flash",
    "test_strategic-deception_glm_4_7_flash",
    "test_strategic-deception_benign_glm_4_7_flash",
    "test_self-preservation_glm_4_7_flash",
    "test_self-preservation_benign_glm_4_7_flash",
    "test_instructed-strategic-sandbagging_glm_4_7_flash",
    "test_instructed-strategic-sandbagging_benign_glm_4_7_flash",
    "test_instructed-covert-code-sabotage_glm_4_7_flash",
    "test_instructed-covert-code-sabotage_benign_glm_4_7_flash",
    "bloom_rollout", "bloom", "mask",
    "sycophancy_feedback_filtered", "sycophancy_answer_filtered",
]


def operating_points(recs):
    n_pos = sum(1 for r in recs.values() if r["gt_pos"])
    n_neg = sum(r["neg_total"] for r in recs.values())
    pf = defaultdict(int)
    nf = defaultdict(int)
    for r in recs.values():
        if r["gt_pos"]:
            for x, f in r["pos_flagged"].items():
                if f:
                    pf[x] += 1
        for x, c in r["neg_flagged"].items():
            nf[x] += c
    return [(nf[x] / n_neg, pf[x] / n_pos) for x in CASCADE_FPRS]


def trap_auc(pts):
    p = sorted(set(pts))
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(p, p[1:]))


def main():
    cascade_pt = load_cascade_per_turn_rows()
    probe_pts = operating_points(
        build_records_discrete("probe", cascade_pt, set(TEST_15)))
    casc_pts = operating_points(
        build_records_discrete("cascade", cascade_pt, set(TEST_15)))

    probe_curve = sorted(set([(0, 0)] + probe_pts + [(1, 1)]))
    casc_curve = sorted(set([(0, 0)] + casc_pts + [(1, 1)]))
    probe_auc = trap_auc(probe_curve)
    casc_auc = trap_auc(casc_curve)
    casc_max_fpr = max(x for x, _ in casc_pts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- Left: full ROC [0, 1] x [0, 1] -----------------------------------
    ax = axes[0]
    px, py = zip(*probe_curve)
    cx, cy = zip(*casc_curve)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1,
            label="Chance")
    ax.plot(px, py, "-o", color="#1f77b4", linewidth=2, markersize=4,
            label=f"Probe (AUC={probe_auc:.3f})")
    cx_meas = [x for x, _ in casc_pts] + [casc_max_fpr]
    cy_meas = [y for _, y in casc_pts] + [casc_pts[-1][1]]
    cx_extrap = [casc_max_fpr, 1.0]
    cy_extrap = [casc_pts[-1][1], 1.0]
    # measured cascade segments
    ax.plot([0] + [x for x, _ in casc_pts], [0] + [y for _, y in casc_pts],
            "-o", color="#d62728", linewidth=2, markersize=4,
            label=f"Cascade (AUC={casc_auc:.3f})")
    # extrapolated cascade tail (dashed)
    ax.plot(cx_extrap, cy_extrap, "--", color="#d62728", linewidth=2,
            alpha=0.6, label="Cascade tail (interpolation only)")
    ax.fill_between(cx_extrap, cy_extrap, [casc_pts[-1][1]] * 2,
                    color="#d62728", alpha=0.08)
    ax.axvline(casc_max_fpr, color="#d62728", linestyle=":", linewidth=1,
               alpha=0.6)
    ax.text(casc_max_fpr + 0.01, 0.05,
            f"Cascade ceiling\n(FPR={casc_max_fpr:.3f})",
            fontsize=9, color="#d62728", va="bottom")
    ax.set_xlabel("Turn-level FPR")
    ax.set_ylabel("Transcript-level TPR")
    ax.set_title("Full ROC (0–1)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # --- Right: zoom to FPR ∈ [0, 0.2] -------------------------------------
    ax = axes[1]
    ax.plot([0, 0.2], [0, 0.2], color="gray", linestyle=":", linewidth=1)
    ax.plot(px, py, "-o", color="#1f77b4", linewidth=2, markersize=5,
            label=f"Probe")
    ax.plot([0] + [x for x, _ in casc_pts], [0] + [y for _, y in casc_pts],
            "-o", color="#d62728", linewidth=2, markersize=5,
            label=f"Cascade")
    ax.axvline(casc_max_fpr, color="#d62728", linestyle=":", linewidth=1,
               alpha=0.6)
    # mark some operating points
    fpr_marks = [0.01, 0.05, 0.10]

    def tpr_at(curve, fpr):
        p = sorted(set(curve))
        for (x0, y0), (x1, y1) in zip(p, p[1:]):
            if x0 <= fpr <= x1:
                if x1 == x0:
                    return y1
                return y0 + (fpr - x0) / (x1 - x0) * (y1 - y0)
        return None

    for fpr in fpr_marks:
        tp = tpr_at(probe_curve, fpr)
        tc = tpr_at(casc_curve, fpr)
        ax.annotate("", xy=(fpr, tc), xytext=(fpr, tp),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                    linewidth=1.5))
        ax.text(fpr + 0.003, (tp + tc) / 2,
                f"+{tc - tp:.2f}", color="#2ca02c", fontsize=9, va="center")
    ax.set_xlabel("Turn-level FPR")
    ax.set_ylabel("Transcript-level TPR")
    ax.set_title("Zoom: FPR ∈ [0, 0.2]")
    ax.set_xlim(-0.005, 0.2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "Probe vs Probes+Haiku Cascade — pooled across 15 paper test behaviors",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "results" / "v4_v2_6_combined_v2_span" / "roc_probe_vs_cascade_test15.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
