"""
Compute PCA on neutral activations and check cosine similarity with trained probes.

Loads per-layer activation tensors from extract_activations.py, computes PCA
using randomized SVD (torch.pca_lowrank), saves the top principal components,
and reports how much each trained probe direction aligns with the top PCs.

High cosine similarity with top PCs indicates the probe may be capturing generic
variation rather than the target concept.

Usage:
    python -m probe.neutral.pca_analysis
    python -m probe.neutral.pca_analysis --n-components 50 --probes-dir probe/probes/v2_3_gen_prompt_v2_span
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.detectors import LogisticRegressionDetector

DEFAULT_LAYERS = [27, 28, 29, 30]


def compute_pca(
    activations: torch.Tensor,
    n_components: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA on activations using randomized SVD.

    Args:
        activations: [n_tokens, emb_dim] float16/float32
        n_components: number of principal components

    Returns:
        components: [n_components, emb_dim] orthonormal PC directions
        explained_variance: [n_components] variance explained by each PC
        mean: [emb_dim] data mean
    """
    acts = activations.float()
    mean = acts.mean(dim=0)

    # torch.pca_lowrank centers internally when center=True (default)
    U, S, V = torch.pca_lowrank(acts, q=n_components, center=True, niter=5)

    # V: [emb_dim, n_components] — columns are PC directions
    components = V.T  # [n_components, emb_dim]

    # Explained variance: S^2 / (n - 1)
    explained_variance = (S**2) / (acts.shape[0] - 1)

    return components, explained_variance, mean


def get_probe_raw_direction(detector: LogisticRegressionDetector) -> torch.Tensor:
    """
    Get the probe direction in raw (unnormalized) activation space.

    The detector operates in normalized space: score = ((acts - mean) / scale) @ direction.
    The effective direction in raw space is direction / scale.
    """
    direction = detector.directions[0].float()  # [emb_dim]
    if detector.scaler_scale is not None:
        scale = detector.scaler_scale[0].float()
        return direction / scale.clamp(min=1e-8)
    return direction


def analyze_probe_pca_overlap(
    probe_dir: Path,
    components: torch.Tensor,
    layer: int,
) -> dict | None:
    """Compute overlap between a probe direction and PCA components."""
    layer_dir = probe_dir / "span" / f"layer{layer}"
    if not (layer_dir / "detector.pt").exists():
        return None

    detector = LogisticRegressionDetector.load(layer_dir / "detector.pt")
    raw_dir = get_probe_raw_direction(detector)
    raw_dir_norm = F.normalize(raw_dir.unsqueeze(0), dim=1)[0]

    # Cosine similarity with each PC
    cos_sims = (components.float() @ raw_dir_norm.float()).abs()
    max_sim, max_idx = cos_sims.max(dim=0)

    # Cumulative variance captured by projection onto top-k PCs
    # This measures how much of the probe direction lies in the generic subspace
    projections_sq = cos_sims**2
    top5_overlap = projections_sq[:5].sum().sqrt().item()
    top10_overlap = projections_sq[:10].sum().sqrt().item()
    top20_overlap = projections_sq[:20].sum().sqrt().item()

    return {
        "max_cos_sim": max_sim.item(),
        "max_pc_idx": max_idx.item(),
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "top20_overlap": top20_overlap,
        "all_cos_sims": cos_sims.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute PCA on neutral activations and analyze probe overlap"
    )
    parser.add_argument(
        "--activations-dir", type=str, default=None,
        help="Directory with per-layer activation .pt files",
    )
    parser.add_argument(
        "--probes-dir", type=str, default=None,
        help="Directory with trained probes (default: probe/probes/v2_3_gen_prompt_v2_span)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save PCA results (default: probe/data/neutral/pca/)",
    )
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data" / "neutral"
    activations_dir = (
        Path(args.activations_dir) if args.activations_dir
        else base_dir / "activations"
    )
    probes_dir = (
        Path(args.probes_dir) if args.probes_dir
        else Path(__file__).parent.parent / "probes" / "v2_3_gen_prompt_v2_span"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else base_dir / "pca"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for layer in args.layers:
        print(f"\n{'=' * 70}")
        print(f"Layer {layer}")
        print(f"{'=' * 70}")

        # Load activations
        act_path = activations_dir / f"layer{layer}.pt"
        if not act_path.exists():
            print(f"  Activations not found at {act_path}, skipping.")
            continue
        acts = torch.load(act_path, weights_only=True)
        print(f"Loaded activations: {acts.shape} ({acts.dtype})")

        # Compute PCA
        print(f"Computing PCA with {args.n_components} components...")
        components, variance, mean = compute_pca(acts, args.n_components)
        total_var = acts.float().var(dim=0).sum().item()
        explained_ratio = variance.sum().item() / total_var if total_var > 0 else 0
        print(f"  Top {args.n_components} PCs explain {explained_ratio:.1%} of total variance")
        print(f"  Top-5 explained variance: {variance[:5].tolist()}")

        # Save PCA results
        pca_path = output_dir / f"layer{layer}_pca.pt"
        torch.save({
            "components": components,       # [n_components, emb_dim]
            "explained_variance": variance, # [n_components]
            "mean": mean,                   # [emb_dim]
            "n_tokens": acts.shape[0],
            "total_variance": total_var,
        }, pca_path)
        print(f"  Saved PCA to {pca_path}")

        # Analyze probe overlap
        print(f"\nProbe overlap with top PCs (probes from {probes_dir.name}):")
        header = (
            f"  {'Indicator':<45} {'Max|cos|':>9} {'PC#':>4}"
            f"  {'Top5':>6} {'Top10':>6} {'Top20':>6}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for indicator_dir in sorted(probes_dir.iterdir()):
            if not indicator_dir.is_dir():
                continue

            result = analyze_probe_pca_overlap(indicator_dir, components, layer)
            if result is None:
                continue

            name = indicator_dir.name
            if len(name) > 44:
                name = name[:41] + "..."
            print(
                f"  {name:<45} {result['max_cos_sim']:>9.4f} {result['max_pc_idx']:>4}"
                f"  {result['top5_overlap']:>6.3f} {result['top10_overlap']:>6.3f}"
                f"  {result['top20_overlap']:>6.3f}"
            )

            key = f"{indicator_dir.name}/layer{layer}"
            all_results[key] = {
                "indicator": indicator_dir.name,
                "layer": layer,
                **{k: v for k, v in result.items() if k != "all_cos_sims"},
            }

    # Save summary
    summary_path = output_dir / "pca_probe_overlap.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")
    print(f"PCA components saved to {output_dir}")


if __name__ == "__main__":
    main()
