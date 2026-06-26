"""
Compare coarse vs finegrain indicator results for paired behaviors.

Auto-discovers pairs in indicator_results/:
  - {xxx}_coarse  <->  {xxx}_finegrain
  - {xxx}         <->  {xxx}_finegrain  (when no _coarse variant exists)

For each pair, computes and compares:
  1. Average consistency across runs (binary agreement, Fleiss' kappa,
     indicator-name Jaccard, sentence Jaccard)
  2. Average detected indicators per trajectory (across runs and filtered)
  3. Detection rate (fraction of trajectories with any indicators)

Usage:
    python bloom/compare_coarse_finegrain.py
"""

import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

INDICATOR_RESULTS_DIR = Path(__file__).parent / "indicator_results"

# Reuse helpers from evaluate_indicator_consistency
sys.path.insert(0, str(Path(__file__).parent.parent))
from bloom.evaluate_indicator_consistency import (
    load_run,
    load_filtered,
    get_binary,
    get_indicator_names,
    get_sentences,
    fleiss_kappa,
    cohen_kappa,
    jaccard,
)


# ──────────────────────────────────────────────────────────────────────
# Pair discovery
# ──────────────────────────────────────────────────────────────────────

def discover_pairs(base_dir: Path) -> list[tuple[str, str]]:
    """Find (coarse, finegrain) directory pairs.

    Pairing rules:
      - {xxx}_coarse + {xxx}_finegrain
      - {xxx} + {xxx}_finegrain (when {xxx}_coarse doesn't exist)
    """
    dirs = sorted(
        d for d in os.listdir(base_dir)
        if (base_dir / d).is_dir()
    )
    dir_set = set(dirs)
    pairs = []
    seen = set()

    for d in dirs:
        if d.endswith("_finegrain"):
            base = d.rsplit("_finegrain", 1)[0]
            coarse = base + "_coarse"
            if coarse in dir_set:
                pairs.append((coarse, d))
                seen.update([coarse, d])
            elif base in dir_set:
                pairs.append((base, d))
                seen.update([base, d])

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Per-directory metrics
# ──────────────────────────────────────────────────────────────────────

def compute_dir_metrics(results_dir: Path) -> dict:
    """Compute consistency and detection metrics for a single indicator dir."""
    metrics = {}

    # Load runs
    run_files = sorted(results_dir.glob("rollout_run*.json"))
    n_runs = len(run_files)
    if n_runs == 0:
        return {"error": "no run files"}

    runs = [load_run(rf) for rf in run_files]

    # Load filtered result
    filtered_path = results_dir / "rollout_aggregated.json"
    if not filtered_path.exists():
        filtered_path = results_dir / "rollout_filtered.json"
    if not filtered_path.exists():
        return {"error": "no filtered file"}
    filtered = load_filtered(filtered_path)

    all_variations = sorted(set().union(*(r.keys() for r in runs), filtered.keys()))
    n_vars = len(all_variations)

    metrics["n_variations"] = n_vars
    metrics["n_runs"] = n_runs

    # ── Detection stats ──
    # Per-run: average indicators per trajectory and detection rate
    run_avg_indicators = []
    run_detection_rates = []
    for run in runs:
        counts = [len(run.get(v, [])) for v in all_variations]
        binaries = [get_binary(run.get(v, [])) for v in all_variations]
        run_avg_indicators.append(np.mean(counts))
        run_detection_rates.append(np.mean(binaries))

    metrics["avg_indicators_per_traj_runs"] = float(np.mean(run_avg_indicators))
    metrics["std_indicators_per_traj_runs"] = float(np.std(run_avg_indicators))
    metrics["detection_rate_runs"] = float(np.mean(run_detection_rates))

    # Filtered: average indicators per trajectory and detection rate
    filt_counts = [len(filtered.get(v, [])) for v in all_variations]
    filt_binaries = [get_binary(filtered.get(v, [])) for v in all_variations]
    metrics["avg_indicators_per_traj_filtered"] = float(np.mean(filt_counts))
    metrics["detection_rate_filtered"] = float(np.mean(filt_binaries))

    # Total unique indicator types across all runs
    all_indicator_names = set()
    for run in runs:
        for v in all_variations:
            all_indicator_names.update(get_indicator_names(run.get(v, [])))
    metrics["n_unique_indicator_types"] = len(all_indicator_names)

    # ── Consistency stats (need >= 2 runs) ──
    if n_runs >= 2:
        # Binary agreement
        binary_per_run = []
        for run in runs:
            binary_per_run.append([get_binary(run.get(v, [])) for v in all_variations])

        full_agree = sum(
            len(set(binary_per_run[i][j] for i in range(n_runs))) == 1
            for j in range(n_vars)
        ) / n_vars
        metrics["binary_agreement_rate"] = float(full_agree)

        # Fleiss' kappa
        ratings = np.zeros((n_vars, 2), dtype=float)
        for j in range(n_vars):
            for i in range(n_runs):
                ratings[j, binary_per_run[i][j]] += 1
        metrics["fleiss_kappa"] = float(fleiss_kappa(ratings))

        # Pairwise Cohen's kappa
        pairwise_ck = [
            cohen_kappa(binary_per_run[i], binary_per_run[j])
            for i, j in combinations(range(n_runs), 2)
        ]
        metrics["mean_cohen_kappa"] = float(np.mean(pairwise_ck))

        # Indicator-name Jaccard
        name_jacs = []
        for v in all_variations:
            name_sets = [get_indicator_names(run.get(v, [])) for run in runs]
            for i, j in combinations(range(n_runs), 2):
                name_jacs.append(jaccard(name_sets[i], name_sets[j]))
        metrics["mean_indicator_name_jaccard"] = float(np.mean(name_jacs))

        # Sentence Jaccard
        sent_jacs = []
        for v in all_variations:
            sent_sets = [get_sentences(run.get(v, [])) for run in runs]
            for i, j in combinations(range(n_runs), 2):
                sent_jacs.append(jaccard(sent_sets[i], sent_sets[j]))
        metrics["mean_sentence_jaccard"] = float(np.mean(sent_jacs))
    else:
        metrics["binary_agreement_rate"] = None
        metrics["fleiss_kappa"] = None
        metrics["mean_cohen_kappa"] = None
        metrics["mean_indicator_name_jaccard"] = None
        metrics["mean_sentence_jaccard"] = None

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────

def fmt(val, width=10, decimals=3):
    """Format a metric value for table display."""
    if val is None:
        return f"{'N/A':^{width}}"
    if isinstance(val, float):
        return f"{val:^{width}.{decimals}f}"
    return f"{str(val):^{width}}"


def print_comparison_table(pairs: list[tuple[str, str]], all_metrics: dict[str, dict]):
    """Print a side-by-side comparison table for all pairs."""
    metric_keys = [
        ("avg_indicators_per_traj_runs", "Avg ind/traj (runs)"),
        ("avg_indicators_per_traj_filtered", "Avg ind/traj (filtered)"),
        ("detection_rate_runs", "Detection rate (runs)"),
        ("detection_rate_filtered", "Detection rate (filtered)"),
        ("n_unique_indicator_types", "Unique indicator types"),
        ("binary_agreement_rate", "Binary agreement"),
        ("fleiss_kappa", "Fleiss' kappa"),
        ("mean_cohen_kappa", "Mean Cohen's kappa"),
        ("mean_indicator_name_jaccard", "Indicator name Jaccard"),
        ("mean_sentence_jaccard", "Sentence Jaccard"),
    ]

    for coarse_dir, fine_dir in pairs:
        # Derive a readable behavior name
        if coarse_dir.endswith("_coarse"):
            behavior = coarse_dir.rsplit("_coarse", 1)[0]
        else:
            behavior = coarse_dir
        coarse_label = "coarse" if coarse_dir.endswith("_coarse") else "base"

        m_c = all_metrics.get(coarse_dir, {})
        m_f = all_metrics.get(fine_dir, {})

        if "error" in m_c or "error" in m_f:
            print(f"\n{'='*70}")
            print(f"PAIR: {behavior}  (SKIPPED — missing data)")
            if "error" in m_c:
                print(f"  {coarse_dir}: {m_c['error']}")
            if "error" in m_f:
                print(f"  {fine_dir}: {m_f['error']}")
            continue

        print(f"\n{'='*70}")
        print(f"PAIR: {behavior}")
        print(f"  {coarse_label}: {coarse_dir}")
        print(f"  finegrain: {fine_dir}")
        print(f"{'='*70}")

        col_w = 14
        label_w = 28
        header = f"{'Metric':<{label_w}} {coarse_label:^{col_w}} {'finegrain':^{col_w}} {'delta':^{col_w}}"
        print(header)
        print("-" * len(header))

        for key, label in metric_keys:
            v_c = m_c.get(key)
            v_f = m_f.get(key)

            if v_c is not None and v_f is not None and isinstance(v_c, (int, float)) and isinstance(v_f, (int, float)):
                delta = v_f - v_c
                delta_str = f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+d}"
            else:
                delta_str = "---"

            print(f"{label:<{label_w}} {fmt(v_c, col_w)} {fmt(v_f, col_w)} {delta_str:^{col_w}}")


def print_summary_grid(pairs: list[tuple[str, str]], all_metrics: dict[str, dict]):
    """Print a compact grid summarizing key metrics across all pairs."""
    print(f"\n\n{'='*90}")
    print("SUMMARY GRID")
    print(f"{'='*90}")

    # Key metrics to show in the grid
    grid_metrics = [
        ("detection_rate_filtered", "Det.Rate(filt)"),
        ("avg_indicators_per_traj_filtered", "AvgInd(filt)"),
        ("binary_agreement_rate", "BinAgree"),
        ("fleiss_kappa", "Fleiss-K"),
        ("mean_sentence_jaccard", "SentJacc"),
    ]

    col_w = 16
    name_w = 30

    # Header
    header = f"{'Behavior':<{name_w}} {'Type':^8}"
    for _, label in grid_metrics:
        header += f" {label:^{col_w}}"
    print(header)
    print("-" * len(header))

    for coarse_dir, fine_dir in pairs:
        if coarse_dir.endswith("_coarse"):
            behavior = coarse_dir.rsplit("_coarse", 1)[0]
        else:
            behavior = coarse_dir
        coarse_label = "coarse" if coarse_dir.endswith("_coarse") else "base"

        m_c = all_metrics.get(coarse_dir, {})
        m_f = all_metrics.get(fine_dir, {})

        if "error" in m_c or "error" in m_f:
            continue

        # Coarse row
        row_c = f"{behavior:<{name_w}} {coarse_label:^8}"
        for key, _ in grid_metrics:
            row_c += f" {fmt(m_c.get(key), col_w)}"
        print(row_c)

        # Finegrain row
        row_f = f"{'':<{name_w}} {'fine':^8}"
        for key, _ in grid_metrics:
            row_f += f" {fmt(m_f.get(key), col_w)}"
        print(row_f)

        # Delta row
        row_d = f"{'':<{name_w}} {'delta':^8}"
        for key, _ in grid_metrics:
            v_c = m_c.get(key)
            v_f = m_f.get(key)
            if v_c is not None and v_f is not None and isinstance(v_c, (int, float)) and isinstance(v_f, (int, float)):
                delta = v_f - v_c
                row_d += f" {f'{delta:+.3f}':^{col_w}}"
            else:
                row_d += f" {'---':^{col_w}}"
        print(row_d)
        print()


def main():
    pairs = discover_pairs(INDICATOR_RESULTS_DIR)
    if not pairs:
        print("No coarse/finegrain pairs found.")
        return

    print(f"Found {len(pairs)} pairs:")
    for c, f in pairs:
        print(f"  {c}  <->  {f}")

    # Compute metrics for all directories in the pairs
    all_dirs = set()
    for c, f in pairs:
        all_dirs.add(c)
        all_dirs.add(f)

    all_metrics = {}
    for d in sorted(all_dirs):
        print(f"\nComputing metrics for {d}...")
        all_metrics[d] = compute_dir_metrics(INDICATOR_RESULTS_DIR / d)

    # Print detailed comparison for each pair
    print_comparison_table(pairs, all_metrics)

    # Print compact summary grid
    print_summary_grid(pairs, all_metrics)

    # Save results to JSON
    output = {
        "pairs": [{"coarse": c, "finegrain": f} for c, f in pairs],
        "metrics": {k: v for k, v in all_metrics.items()},
    }
    out_path = INDICATOR_RESULTS_DIR / "coarse_vs_finegrain_comparison.json"
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
