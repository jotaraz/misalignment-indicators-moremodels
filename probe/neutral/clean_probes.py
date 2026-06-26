"""
Clean probe directions by projecting out top PCA components from neutral data.

For each probe, converts the direction to raw activation space, removes the
projection onto the top-k neutral PCA components (generic variance directions),
then converts back. This yields a "cleaned" probe that is more specific to the
target concept.

Cleaned probes are saved in the same format as the originals (detector.pt, cfg.yaml,
training_meta.json) for drop-in use with probe_eval.

Usage:
    python -m probe.neutral.clean_probes
    python -m probe.neutral.clean_probes --n-components 10
    python -m probe.neutral.clean_probes --probes-dir probe/probes/v2_3_gen_prompt_v2_span \
        --output-dir probe/probes/v2_3_gen_prompt_v2_span_clean
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.detectors import LogisticRegressionDetector
from probe.train import _compute_val_thresholds_turn

DEFAULT_LAYERS = [27, 28, 29, 30]


def project_out_components(
    direction: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    """
    Project direction onto the subspace orthogonal to the given components.

    Args:
        direction: [emb_dim] — the vector to clean
        components: [k, emb_dim] — orthonormal directions to remove

    Returns:
        cleaned direction [emb_dim]
    """
    # projections[i] = direction . components[i]
    projections = components @ direction  # [k]
    # reconstruction = sum_i (proj_i * components[i])
    reconstruction = projections @ components  # [emb_dim]
    return direction - reconstruction


def clean_detector(
    detector: LogisticRegressionDetector,
    components: torch.Tensor,
) -> tuple[LogisticRegressionDetector, dict]:
    """
    Clean a detector's direction by projecting out PCA components.

    Operates in raw activation space:
    1. Convert direction to raw space: raw_dir = direction / scaler_scale
    2. Project out top PCs: clean_raw = raw_dir - proj(raw_dir, PCs)
    3. Convert back: clean_direction = clean_raw * scaler_scale

    Returns (cleaned_detector, stats_dict).
    """
    direction = detector.directions[0].float()  # [emb_dim]
    components = F.normalize(components.float(), dim=1)

    # Convert to raw activation space
    has_scaler = detector.scaler_scale is not None
    if has_scaler:
        scale = detector.scaler_scale[0].float().clamp(min=1e-8)
        raw_direction = direction / scale
    else:
        raw_direction = direction

    # Project out PCs
    clean_raw = project_out_components(raw_direction, components)

    # Convert back to normalized space
    if has_scaler:
        clean_direction = clean_raw * scale
    else:
        clean_direction = clean_raw

    # Compute stats
    cos_orig_clean = F.cosine_similarity(
        direction.unsqueeze(0), clean_direction.unsqueeze(0)
    ).item()
    norm_ratio = clean_direction.norm().item() / direction.norm().clamp(min=1e-10).item()
    removed_norm = (direction - clean_direction).norm().item()

    # Update detector (create a new one to avoid modifying the original)
    cleaned = LogisticRegressionDetector(
        layers=detector.layers,
        reg_coeff=detector.reg_coeff,
        normalize=detector.normalize,
    )
    cleaned.directions = clean_direction.unsqueeze(0)
    cleaned.scaler_mean = detector.scaler_mean
    cleaned.scaler_scale = detector.scaler_scale

    stats = {
        "cos_orig_clean": cos_orig_clean,
        "norm_ratio": norm_ratio,
        "removed_norm": removed_norm,
        "orig_norm": direction.norm().item(),
    }
    return cleaned, stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean probe directions using neutral PCA components"
    )
    parser.add_argument(
        "--pca-dir", type=str, default=None,
        help="Directory with PCA .pt files (default: probe/data/neutral/pca/)",
    )
    parser.add_argument(
        "--probes-dir", type=str, default=None,
        help="Directory with trained probes (default: probe/probes/v2_3_gen_prompt_v2_span)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for cleaned probes (default: probe/probes/v2_3_gen_prompt_v2_span_clean)",
    )
    parser.add_argument(
        "--n-components", type=int, default=10,
        help="Number of PCs to project out (default: 10)",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    pca_dir = (
        Path(args.pca_dir) if args.pca_dir
        else base_dir / "data" / "neutral" / "pca"
    )
    probes_dir = (
        Path(args.probes_dir) if args.probes_dir
        else base_dir / "probes" / "v2_3_gen_prompt_v2_span"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else base_dir / "probes" / "v2_3_gen_prompt_v2_span_clean"
    )

    all_stats: list[dict] = []

    for layer in args.layers:
        print(f"\n{'=' * 70}")
        print(f"Layer {layer}: removing top {args.n_components} PCs")
        print(f"{'=' * 70}")

        # Load PCA
        pca_path = pca_dir / f"layer{layer}_pca.pt"
        if not pca_path.exists():
            print(f"  PCA not found at {pca_path}, skipping.")
            continue
        pca_data = torch.load(pca_path, weights_only=True)
        components = pca_data["components"][: args.n_components]  # [k, emb_dim]

        header = f"  {'Indicator':<45} {'cos(o,c)':>9} {'norm%':>7} {'removed':>9} {'val_F1':>7} {'val_Acc':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for indicator_dir in sorted(probes_dir.iterdir()):
            if not indicator_dir.is_dir():
                continue

            src_layer_dir = indicator_dir / "span" / f"layer{layer}"
            if not (src_layer_dir / "detector.pt").exists():
                continue

            # Load and clean
            detector = LogisticRegressionDetector.load(src_layer_dir / "detector.pt")
            cleaned, stats = clean_detector(detector, components)

            name = indicator_dir.name
            if len(name) > 44:
                name = name[:41] + "..."
            val_f1_str = f"{stats.get('val_f1', float('nan')):>7.3f}" if "val_f1" in stats else "    N/A"
            val_acc_str = f"{stats.get('val_acc', float('nan')):>7.3f}" if "val_acc" in stats else "    N/A"
            print(
                f"  {name:<45} {stats['cos_orig_clean']:>9.4f}"
                f" {stats['norm_ratio']:>6.1%} {stats['removed_norm']:>9.4f}"
                f" {val_f1_str} {val_acc_str}"
            )

            # Save cleaned detector
            dst_layer_dir = output_dir / indicator_dir.name / "span" / f"layer{layer}"
            dst_layer_dir.mkdir(parents=True, exist_ok=True)
            cleaned.save(dst_layer_dir / "detector.pt")

            # Copy cfg.yaml
            cfg_src = src_layer_dir / "cfg.yaml"
            if cfg_src.exists():
                shutil.copy2(cfg_src, dst_layer_dir / "cfg.yaml")

            # Recompute val thresholds if val_acts.pt exists
            val_acts_path = src_layer_dir / "val_acts.pt"
            new_val_thresholds = None
            if val_acts_path.exists():
                val_data = torch.load(val_acts_path, weights_only=True)
                # val_acts are [n_tokens, 1, n_features] — layer_idx=0
                new_val_thresholds = _compute_val_thresholds_turn(
                    cleaned, val_data["pos"], val_data["neg"], layer_idx=0,
                )
                stats["val_f1"] = new_val_thresholds["best_f1"]
                stats["val_acc"] = new_val_thresholds["best_accuracy"]
                # Copy val_acts.pt to output so future re-cleaning can use it
                shutil.copy2(val_acts_path, dst_layer_dir / "val_acts.pt")

            # Copy and update training_meta.json with new val thresholds
            meta_src = src_layer_dir / "training_meta.json"
            if meta_src.exists():
                with open(meta_src) as f:
                    meta = json.load(f)
                if new_val_thresholds is not None:
                    meta["val_thresholds_pre_cleaning"] = meta.get("val_thresholds")
                    meta["val_thresholds"] = new_val_thresholds
                with open(dst_layer_dir / "training_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

            all_stats.append({
                "indicator": indicator_dir.name,
                "layer": layer,
                **stats,
            })

    # Save cleaning metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_meta = {
        "n_components_removed": args.n_components,
        "pca_dir": str(pca_dir),
        "source_probes_dir": str(probes_dir),
        "layers": args.layers,
        "per_probe_stats": all_stats,
    }
    with open(output_dir / "cleaning_meta.json", "w") as f:
        json.dump(clean_meta, f, indent=2)

    print(f"\nCleaned probes saved to {output_dir}")
    print(f"Cleaning metadata saved to {output_dir / 'cleaning_meta.json'}")

    # Summary
    if all_stats:
        avg_cos = sum(s["cos_orig_clean"] for s in all_stats) / len(all_stats)
        avg_norm = sum(s["norm_ratio"] for s in all_stats) / len(all_stats)
        print(f"\nSummary across {len(all_stats)} probe-layer pairs:")
        print(f"  Avg cos(original, cleaned): {avg_cos:.4f}")
        print(f"  Avg norm retention: {avg_norm:.1%}")


if __name__ == "__main__":
    main()
