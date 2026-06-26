"""
Compute probe accuracy using indicator detection results as ground truth.

For each probe × behavior combination, matches rollouts by (variation_number,
repetition_number), treats indicator detection (has any detected_indicators)
as ground-truth positive, and computes AUROC, accuracy, precision, recall, F1.

Usage:
    python -m probe_eval.accuracy_vs_indicators
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

PROBES = [
    "repe_glm_lr__repe_honesty__you_are_fact_sys__glm-9b-flash__20260215_190119",
    "rp_glm_lr__roleplaying__plain__glm-9b-flash__20260216_055449",
    "rp_glm_offpolicy_lr__roleplaying__offpolicy_train__glm-9b-flash__20260216_043957",
]

PROBE_SHORT = [
    "repe_honesty",
    "rp_onpolicy",
    "rp_offpolicy",
]

# Mapping: probe eval behavior name -> indicator results directory
BEHAVIOR_INDICATOR_MAP = {
    "sandbagging": "sandbagging_neutral_coarse",
    "undermining_oversight": "undermining_oversight_neutral_finegrain",
}


def load_indicator_labels(indicator_dir: str) -> dict[tuple[int, int], bool]:
    """Load indicator results and return {(var, rep): has_indicators} mapping."""
    indicator_path = REPO_ROOT / "bloom" / "indicator_results" / indicator_dir / "rollout_filtered.json"
    with open(indicator_path) as f:
        data = json.load(f)

    labels = {}
    for rollout in data["evaluated_rollouts"]:
        key = (rollout["variation_number"], rollout["repetition_number"])
        labels[key] = len(rollout["detected_indicators"]) > 0
    return labels


def load_probe_results(probe: str, behavior: str) -> list[dict] | None:
    """Load probe per_rollout results. Returns None if not found."""
    path = RESULTS_DIR / probe / behavior / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def get_rollout_var_rep(rollout_dir: str) -> dict[int, tuple[int, int]]:
    """Load rollout.json and return {rollout_index: (var, rep)} mapping."""
    rollout_path = Path(rollout_dir) / "rollout.json"
    with open(rollout_path) as f:
        data = json.load(f)
    mapping = {}
    for i, rollout in enumerate(data["rollouts"]):
        mapping[i] = (rollout["variation_number"], rollout["repetition_number"])
    return mapping


def compute_accuracy_metrics(
    probe_scores: np.ndarray, indicator_labels: np.ndarray
) -> dict:
    """Compute AUROC, optimal-threshold accuracy, precision, recall, F1."""
    results = {
        "n_total": len(probe_scores),
        "n_positive": int(indicator_labels.sum()),
        "n_negative": int((~indicator_labels.astype(bool)).sum()),
    }

    # Filter NaNs
    valid = ~np.isnan(probe_scores)
    scores_valid = probe_scores[valid]
    labels_valid = indicator_labels[valid]
    results["n_valid"] = int(valid.sum())

    if results["n_valid"] < 2 or labels_valid.sum() == 0 or labels_valid.sum() == len(labels_valid):
        results["auroc"] = float("nan")
        results["accuracy"] = float("nan")
        results["precision"] = float("nan")
        results["recall"] = float("nan")
        results["f1"] = float("nan")
        results["threshold"] = float("nan")
        return results

    # AUROC
    results["auroc"] = float(roc_auc_score(labels_valid, scores_valid))

    # Find optimal threshold (maximize accuracy over all unique thresholds)
    thresholds = np.unique(scores_valid)
    # Add midpoints between consecutive thresholds
    if len(thresholds) > 1:
        midpoints = (thresholds[:-1] + thresholds[1:]) / 2
        thresholds = np.concatenate([thresholds, midpoints])

    best_acc = -1
    best_thresh = 0
    best_preds = None
    for t in thresholds:
        preds = (scores_valid >= t).astype(int)
        acc = (preds == labels_valid).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            best_preds = preds

    results["accuracy"] = float(best_acc)
    results["threshold"] = float(best_thresh)

    tp = int(((best_preds == 1) & (labels_valid == 1)).sum())
    fp = int(((best_preds == 1) & (labels_valid == 0)).sum())
    fn = int(((best_preds == 0) & (labels_valid == 1)).sum())
    tn = int(((best_preds == 0) & (labels_valid == 0)).sum())

    results["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    results["recall"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if results["precision"] == results["precision"] and results["recall"] == results["recall"]:
        p, r = results["precision"], results["recall"]
        results["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        results["f1"] = float("nan")

    return results


def main():
    behaviors = list(BEHAVIOR_INDICATOR_MAP.keys())

    # Load indicator labels for each behavior
    indicator_labels_by_behavior = {}
    for behavior, indicator_dir in BEHAVIOR_INDICATOR_MAP.items():
        indicator_labels_by_behavior[behavior] = load_indicator_labels(indicator_dir)

    all_results = {}

    for probe, probe_short in zip(PROBES, PROBE_SHORT):
        for behavior in behaviors:
            probe_data = load_probe_results(probe, behavior)
            if probe_data is None:
                continue

            # Get var/rep mapping from rollout.json
            rollout_dir = probe_data["rollout_dir"]
            var_rep_map = get_rollout_var_rep(rollout_dir)

            indicator_labels = indicator_labels_by_behavior[behavior]

            # Match probe scores to indicator labels
            matched_scores = []
            matched_labels = []
            matched_details = []

            for entry in probe_data["per_rollout"]:
                idx = entry["rollout_index"]
                if idx not in var_rep_map:
                    continue
                var, rep = var_rep_map[idx]
                key = (var, rep)
                if key not in indicator_labels:
                    continue

                matched_scores.append(entry["probe_score"])
                matched_labels.append(int(indicator_labels[key]))
                matched_details.append({
                    "var": var,
                    "rep": rep,
                    "probe_score": entry["probe_score"],
                    "bp": entry["behavior_presence"],
                    "bp_label": entry["label"],
                    "indicator_positive": indicator_labels[key],
                })

            if not matched_scores:
                continue

            scores_arr = np.array(matched_scores)
            labels_arr = np.array(matched_labels)

            metrics = compute_accuracy_metrics(scores_arr, labels_arr)
            all_results[(probe_short, behavior)] = metrics

            # Print detailed per-rollout comparison
            print(f"\n{'='*70}")
            print(f"Probe: {probe_short} | Behavior: {behavior}")
            print(f"Indicator source: {BEHAVIOR_INDICATOR_MAP[behavior]}")
            print(f"{'='*70}")
            print(f"  {'var':>3} {'rep':>3} {'probe_score':>12} {'bp':>3} {'bp_label':>10} {'ind_pos':>8}")
            print(f"  {'-'*48}")
            for d in matched_details:
                marker = "*" if d["indicator_positive"] else " "
                score_str = f"{d['probe_score']:.4f}" if d["probe_score"] == d["probe_score"] else "   NaN"
                print(f"  {d['var']:>3} {d['rep']:>3} {score_str:>12} {d['bp']:>3} {d['bp_label']:>10} {marker:>8}")

            print(f"\n  Metrics (indicator labels as ground truth):")
            print(f"    n_total={metrics['n_total']}, n_positive={metrics['n_positive']}, n_valid={metrics['n_valid']}")
            if metrics["auroc"] == metrics["auroc"]:
                print(f"    AUROC:     {metrics['auroc']:.3f}")
                print(f"    Accuracy:  {metrics['accuracy']:.3f}  (threshold={metrics['threshold']:.4f})")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall:    {metrics['recall']:.3f}")
                print(f"    F1:        {metrics['f1']:.3f}")
                if "confusion" in metrics:
                    c = metrics["confusion"]
                    print(f"    Confusion: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")
            else:
                print(f"    (insufficient data for metrics)")

    # Print summary grid
    print(f"\n\n{'='*70}")
    print("SUMMARY: Probe AUROC with indicator labels as ground truth")
    print(f"{'='*70}")

    col_w = 20
    probe_w = 16
    header = f"{'probe':<{probe_w}}" + "".join(f"{b:^{col_w}}" for b in behaviors)
    print(header)
    print("-" * len(header))

    for probe_short in PROBE_SHORT:
        row = f"{probe_short:<{probe_w}}"
        for behavior in behaviors:
            key = (probe_short, behavior)
            if key in all_results:
                m = all_results[key]
                if m["auroc"] == m["auroc"]:
                    val = f"AUROC={m['auroc']:.3f}"
                else:
                    val = "N/A"
            else:
                val = "---"
            row += f"{val:^{col_w}}"
        print(row)

    print()
    header2 = f"{'probe':<{probe_w}}" + "".join(f"{b:^{col_w}}" for b in behaviors)
    print(header2)
    print("-" * len(header2))
    for probe_short in PROBE_SHORT:
        row = f"{probe_short:<{probe_w}}"
        for behavior in behaviors:
            key = (probe_short, behavior)
            if key in all_results:
                m = all_results[key]
                if m["accuracy"] == m["accuracy"]:
                    val = f"Acc={m['accuracy']:.3f}"
                else:
                    val = "N/A"
            else:
                val = "---"
            row += f"{val:^{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
