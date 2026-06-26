"""
Plot pairwise cosine similarity matrix of probe directions.

Usage:
    python -m probe_eval.plot_probe_similarity \
        --probes-dir probe/probes/v4_v2_6_combined_span \
        --layer 27 \
        --output probe_eval/results/probe_cosine_similarity_matrix_v4.png
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ood_misalignment_eval/deception-detection"))
sys.path.insert(0, str(REPO_ROOT))


# Desired behavior order (single-behavior groups first, multi last)
BEHAVIOR_ORDER = [
    "strategic-deception",
    "sycophancy",
    "sabotage",
    "sandbagging",
    "self-preservation",
]


def _slug(name: str) -> str:
    """Convert indicator name to probe folder name (e.g. 'Observer Suspicion Modeling' -> 'observer_suspicion_modeling')."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.lower()).strip("_")
    return s


def get_ordering_from_indicators() -> list[tuple[str, str]]:
    """Return list of (probe_slug, group_label) ordered by behavior groups."""
    from indicators.misalignment_indicators_v2_6 import MISALIGNMENT_INDICATORS_V2
    ordered: list[tuple[str, str]] = []
    seen = set()

    # First: single-behavior indicators, grouped by behavior in BEHAVIOR_ORDER
    for beh in BEHAVIOR_ORDER:
        for ind in MISALIGNMENT_INDICATORS_V2:
            behs = ind.relevant_behaviors or []
            if len(behs) == 1 and behs[0] == beh:
                slug = _slug(ind.name)
                if slug not in seen:
                    ordered.append((slug, beh))
                    seen.add(slug)

    # Then: multi-behavior indicators at the end
    for ind in MISALIGNMENT_INDICATORS_V2:
        behs = ind.relevant_behaviors or []
        if len(behs) > 1:
            slug = _slug(ind.name)
            if slug not in seen:
                ordered.append((slug, "multi"))
                seen.add(slug)

    return ordered


def load_directions(probes_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """Load probe direction vectors from detector.pt files."""
    directions = {}
    for probe_path in sorted(probes_dir.glob(f"*/span/layer{layer}/detector.pt")):
        concept = probe_path.parents[2].name
        with open(probe_path, "rb") as f:
            data = pickle.load(f)
        d = data["directions"].numpy().flatten()
        directions[concept] = d
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--group-by-behavior", action="store_true", default=True,
                        help="Order indicators by their relevant_behaviors (single-behavior first, multi last).")
    args = parser.parse_args()

    probes_dir = Path(args.probes_dir)
    directions = load_directions(probes_dir, args.layer)
    print(f"Loaded {len(directions)} probe directions")

    # Determine ordering
    if args.group_by_behavior:
        ordering = get_ordering_from_indicators()
        # Filter to those that exist in directions
        names = []
        groups = []
        for slug, group in ordering:
            if slug in directions:
                names.append(slug)
                groups.append(group)
        # Append any probes not in the indicator list (fallback)
        leftover = sorted(set(directions) - set(names))
        for slug in leftover:
            names.append(slug)
            groups.append("other")
    else:
        names = sorted(directions.keys())
        groups = ["other"] * len(names)

    # Compute matrix
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v1, v2 = directions[a], directions[b]
            mat[i, j] = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    # Annotate names with group label
    group_colors = {
        "strategic-deception": "#d62728",
        "sycophancy": "#9467bd",
        "sabotage": "#8c564b",
        "sandbagging": "#2ca02c",
        "self-preservation": "#1f77b4",
        "multi": "#7f7f7f",
        "other": "#bcbd22",
    }
    short_names = [n.replace("_", " ") for n in names]

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # Color tick labels by group
    for i, g in enumerate(groups):
        ax.get_xticklabels()[i].set_color(group_colors.get(g, "black"))
        ax.get_yticklabels()[i].set_color(group_colors.get(g, "black"))

    # Draw separators between groups
    boundaries = []
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            boundaries.append(i)
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=0.8, alpha=0.6)
        ax.axvline(b - 0.5, color="black", linewidth=0.8, alpha=0.6)

    # Add cosine values as text
    for i in range(len(names)):
        for j in range(len(names)):
            val = mat[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    # Legend for groups
    handles = []
    seen_groups = []
    for g in groups:
        if g not in seen_groups:
            seen_groups.append(g)
            handles.append(plt.Line2D([0], [0], marker="s", linestyle="",
                                       color=group_colors.get(g, "black"), label=g))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, title="Behavior group")

    title = args.title or f"Probe Cosine Similarity ({probes_dir.name}, layer {args.layer})"
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="Cosine Similarity", pad=0.15)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
