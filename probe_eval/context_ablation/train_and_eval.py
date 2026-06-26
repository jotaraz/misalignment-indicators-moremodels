"""
Step 7: Train probes on context-dependent data and evaluate.

Trains probes in two configurations:
  1. context_only: Trained only on context-dependent pairs
     → Tests if the model's activations contain context-dependent features at all
  2. combined: Trained on original + context-dependent data
     → Tests if one probe can handle both types

Evaluates each on:
  - Context-dependent val set (same span, different context)
  - The three ablation conditions from Step 4

Supports per-indicator parallelism via --indicator flag.

Usage (requires GPU for combined mode with original data):
    python -m probe_eval.context_ablation.train_and_eval
    python -m probe_eval.context_ablation.train_and_eval --indicator action_concealment
    python -m probe_eval.context_ablation.train_and_eval --dataset-type context_only
    python -m probe_eval.context_ablation.train_and_eval --dataset-type combined
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ood_misalignment_eval/deception-detection"))


def train_probe(
    train_acts: np.ndarray,
    train_labels: np.ndarray,
    val_acts: np.ndarray,
    val_labels: np.ndarray,
    reg_coeff: float = 10.0,
) -> dict:
    """Train a logistic regression probe and evaluate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Normalize
    mean = train_acts.mean(axis=0)
    std = train_acts.std(axis=0).clip(min=1e-8)
    train_normed = (train_acts - mean) / std
    val_normed = (val_acts - mean) / std

    clf = LogisticRegression(C=1.0 / reg_coeff, max_iter=1000, solver="lbfgs")
    clf.fit(train_normed, train_labels)

    train_preds = clf.predict_proba(train_normed)[:, 1]
    val_preds = clf.predict_proba(val_normed)[:, 1]

    results = {
        "train_auroc": float(roc_auc_score(train_labels, train_preds)) if len(set(train_labels)) > 1 else None,
        "val_auroc": float(roc_auc_score(val_labels, val_preds)) if len(set(val_labels)) > 1 else None,
        "train_accuracy": float(accuracy_score(train_labels, train_preds > 0.5)),
        "val_accuracy": float(accuracy_score(val_labels, val_preds > 0.5)),
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "n_train_pos": int(train_labels.sum()),
        "n_val_pos": int(val_labels.sum()),
        "weight": clf.coef_[0],
        "bias": float(clf.intercept_[0]),
        "norm_mean": mean,
        "norm_std": std,
    }
    return results


def load_ablation_activations(
    activations_dir: Path,
    probe_scores_path: Path,
    indicator: str,
    layer: int,
) -> dict[str, list[np.ndarray]]:
    """Load pre-extracted activations from step 2, grouped by condition.

    Returns {condition: [mean_span_act_per_rollout, ...]}
    """
    with open(probe_scores_path) as f:
        all_scores = json.load(f)

    layer_key = f"layer{layer}"
    condition_acts = {"no_context": [], "positive_context": [], "negative_context": []}

    n_spans = len(all_scores) // 3
    for si in range(n_spans):
        base = si * 3
        if base + 2 >= len(all_scores):
            break
        if all_scores[base + 1].get("indicator", "") != indicator:
            continue

        for offset, cond in enumerate(["no_context", "positive_context", "negative_context"]):
            act_path = activations_dir / f"activations_{base + offset:04d}.npz"
            if not act_path.exists():
                continue
            acts_data = np.load(act_path)
            if layer_key not in acts_data:
                continue
            span_acts = acts_data[layer_key].mean(axis=0).astype(np.float32)
            condition_acts[cond].append(span_acts)

    return condition_acts


def build_training_from_ablation(
    activations_dir: Path,
    probe_scores_path: Path,
    indicator: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val from pre-extracted ablation activations.

    Positive = positive_context, Negative = negative_context.
    Uses 80/20 split keeping pos/neg pairs together.
    """
    with open(probe_scores_path) as f:
        all_scores = json.load(f)

    layer_key = f"layer{layer}"
    pairs = []  # (pos_act, neg_act) per span

    n_spans = len(all_scores) // 3
    for si in range(n_spans):
        base = si * 3
        if base + 2 >= len(all_scores):
            break
        if all_scores[base + 1].get("indicator", "") != indicator:
            continue

        pos_path = activations_dir / f"activations_{base + 1:04d}.npz"
        neg_path = activations_dir / f"activations_{base + 2:04d}.npz"
        if not pos_path.exists() or not neg_path.exists():
            continue
        pos_data = np.load(pos_path)
        neg_data = np.load(neg_path)
        if layer_key not in pos_data or layer_key not in neg_data:
            continue

        pos_act = pos_data[layer_key].mean(axis=0).astype(np.float32)
        neg_act = neg_data[layer_key].mean(axis=0).astype(np.float32)
        pairs.append((pos_act, neg_act))

    if not pairs:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

    # 80/20 split keeping pairs together
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n_val = max(1, len(pairs) // 5)
    val_idx = set(indices[:n_val])

    train_acts, train_labels = [], []
    val_acts, val_labels = [], []

    for i, (pos_act, neg_act) in enumerate(pairs):
        target = (val_acts, val_labels) if i in val_idx else (train_acts, train_labels)
        target[0].append(pos_act)
        target[1].append(1)
        target[0].append(neg_act)
        target[1].append(0)

    return (
        np.stack(train_acts), np.array(train_labels),
        np.stack(val_acts), np.array(val_labels),
    )


def evaluate_on_ablation(
    weight: np.ndarray,
    bias: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    condition_acts: dict[str, list[np.ndarray]],
) -> dict:
    """Score ablation activations with the trained probe."""
    results = {}
    for cond, acts_list in condition_acts.items():
        if not acts_list:
            results[cond] = {"mean": None, "std": None, "n": 0}
            continue
        acts = np.stack(acts_list)
        normed = (acts - norm_mean) / np.clip(norm_std, 1e-8, None)
        scores = normed @ weight + bias
        results[cond] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": len(scores),
        }
    return results


def process_indicator(indicator: str, ds_type: str, layer: int, reg_coeff: float,
                      training_dir: Path, activations_dir: Path,
                      output_dir: Path) -> dict | None:
    """Train and evaluate a single indicator. Returns result dict or None."""
    probe_scores_path = activations_dir / "probe_scores.json"
    data_file = training_dir / ds_type / f"{indicator}.json"

    if not data_file.exists():
        return None

    with open(data_file) as f:
        data = json.load(f)

    print(f"  {indicator} ({ds_type}):", flush=True)

    if ds_type == "context_only":
        train_acts, train_labels, val_acts, val_labels = build_training_from_ablation(
            activations_dir, probe_scores_path, indicator, layer,
        )
    else:
        train_acts, train_labels, val_acts, val_labels = build_training_from_ablation(
            activations_dir, probe_scores_path, indicator, layer,
        )

    if len(train_acts) == 0 or len(val_acts) == 0:
        print(f"    SKIP: insufficient data", flush=True)
        return None

    print(f"    Train: {len(train_labels)} ({train_labels.sum()} pos), "
          f"Val: {len(val_labels)} ({val_labels.sum()} pos)", flush=True)

    result = train_probe(train_acts, train_labels, val_acts, val_labels, reg_coeff)
    print(f"    Train AUROC: {result['train_auroc']:.3f}, "
          f"Val AUROC: {result['val_auroc']:.3f}", flush=True)

    # Evaluate on ablation conditions
    condition_acts = load_ablation_activations(
        activations_dir, probe_scores_path, indicator, layer,
    )
    ablation_scores = evaluate_on_ablation(
        result["weight"], result["bias"], result["norm_mean"], result["norm_std"],
        condition_acts,
    )
    print(f"    Ablation: pos={ablation_scores['positive_context'].get('mean', 'N/A'):.3f}, "
          f"neg={ablation_scores['negative_context'].get('mean', 'N/A'):.3f}, "
          f"no={ablation_scores['no_context'].get('mean', 'N/A'):.3f}", flush=True)

    # Save probe weights for cross-evaluation
    probe_save_dir = output_dir / "probes" / ds_type
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        probe_save_dir / f"{indicator}_layer{layer}.npz",
        weight=result["weight"],
        bias=np.array([result["bias"]]),
        norm_mean=result["norm_mean"],
        norm_std=result["norm_std"],
    )

    # Clean large arrays before returning to JSON
    result["ablation_condition_scores"] = ablation_scores
    result["weight"] = None
    result["norm_mean"] = None
    result["norm_std"] = None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", choices=["context_only", "combined", "both"],
                        default="both")
    parser.add_argument("--indicator", type=str, nargs="*", default=None,
                        help="Specific indicator(s) to process (default: all)")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--reg-coeff", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "probe_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_dir = DATA_DIR / "training"
    activations_dir = DATA_DIR / "activations"

    dataset_types = ["context_only", "combined"] if args.dataset_type == "both" else [args.dataset_type]

    # Discover indicators
    if args.indicator:
        indicators = args.indicator
    else:
        indicators = set()
        for ds_type in dataset_types:
            ds_dir = training_dir / ds_type
            if ds_dir.exists():
                indicators.update(p.stem for p in ds_dir.glob("*.json"))
        indicators = sorted(indicators)

    print(f"Indicators: {len(indicators)}", flush=True)
    print(f"Dataset types: {dataset_types}", flush=True)

    all_results = {}

    for ds_type in dataset_types:
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {ds_type}", flush=True)
        print(f"{'='*60}", flush=True)

        ds_results = {}
        for indicator in indicators:
            result = process_indicator(
                indicator, ds_type, args.layer, args.reg_coeff,
                training_dir, activations_dir, output_dir,
            )
            if result is not None:
                ds_results[indicator] = result

        all_results[ds_type] = ds_results

    # Save results
    suffix = f"_{'_'.join(args.indicator)}" if args.indicator and len(args.indicator) <= 3 else ""
    results_path = output_dir / f"training_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved -> {results_path}", flush=True)

    # Generate report
    report_lines = ["# Context-Dependent Probe Training Results\n"]

    for ds_type, ds_results in all_results.items():
        report_lines.append(f"## {ds_type}\n")
        report_lines.append("| Indicator | Train AUROC | Val AUROC | No Ctx | Pos Ctx | Neg Ctx | Pos-Neg |")
        report_lines.append("|-----------|-----------|---------|--------|---------|---------|---------|")

        for ind, r in sorted(ds_results.items()):
            t_auc = f"{r['train_auroc']:.3f}" if r.get("train_auroc") else "N/A"
            v_auc = f"{r['val_auroc']:.3f}" if r.get("val_auroc") else "N/A"

            abl = r.get("ablation_condition_scores", {})
            no_ctx = f"{abl.get('no_context', {}).get('mean', 0):.3f}" if abl.get("no_context", {}).get("mean") is not None else "N/A"
            pos_ctx = f"{abl.get('positive_context', {}).get('mean', 0):.3f}" if abl.get("positive_context", {}).get("mean") is not None else "N/A"
            neg_ctx = f"{abl.get('negative_context', {}).get('mean', 0):.3f}" if abl.get("negative_context", {}).get("mean") is not None else "N/A"

            pos_m = abl.get("positive_context", {}).get("mean")
            neg_m = abl.get("negative_context", {}).get("mean")
            delta = f"{pos_m - neg_m:+.3f}" if pos_m is not None and neg_m is not None else "N/A"

            report_lines.append(f"| {ind} | {t_auc} | {v_auc} | {no_ctx} | {pos_ctx} | {neg_ctx} | {delta} |")
        report_lines.append("")

    report_path = output_dir / f"training_report{suffix}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report saved -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
