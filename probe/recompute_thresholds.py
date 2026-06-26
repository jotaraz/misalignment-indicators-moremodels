"""
Recompute val thresholds for already-trained probes using all 4 metric sets.

Loads the original training data, splits train/val, extracts per-sample val
activations, scores them with saved detectors, and computes thresholds for:
  - span_overlap (strict sentence-to-span text overlap)
  - same_turn (sentence in same turn as GT span)
  - turn_level (max sentence score per turn)
  - clean_label (pos = span-overlapping sentences, neg = non-deceptive-turn sentences)

Results are saved back into each layer's training_meta.json under val_thresholds.

Usage:
    python -m probe.recompute_thresholds \
        --probe-dir probe/probes/v2_3_gen_prompt_v2_span_v2 \
        --data-dir probe/data/v2_3_gen_prompt_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

REPO_ROOT = Path(__file__).parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.activations import Activations
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Message, Dialogue

from probe.train import (
    DEFAULT_PADDING,
    TurnSample,
    ValSampleActivations,
    build_turn_samples,
    _split_transcripts_train_val,
    _extract_val_sample_activations,
    _compute_val_thresholds_sentence,
    _print_sentence_val_metrics,
    indicator_name_to_filename,
)


MODEL_NAME = ModelName.GLM_FLASH


def main():
    parser = argparse.ArgumentParser(
        description="Recompute val thresholds for trained probes"
    )
    parser.add_argument(
        "--probe-dir", type=str, required=True,
        help="Root probe directory (e.g. probe/probes/v2_3_gen_prompt_v2_span_v2)",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Training data directory (e.g. probe/data/v2_3_gen_prompt_v2)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Val fraction (must match what was used in training, default: 0.2)",
    )
    parser.add_argument(
        "--indicator", type=str, nargs="+", default=None,
        help="Only recompute for specific indicator(s). Default: all.",
    )
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    data_dir = Path(args.data_dir)

    # Discover indicators from probe dir
    if args.indicator:
        indicator_slugs = [indicator_name_to_filename(name) for name in args.indicator]
    else:
        indicator_slugs = sorted(
            d.name for d in probe_dir.iterdir() if d.is_dir()
        )

    # Load model once
    print("Loading GLM-4.7 Flash model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    for indicator_slug in indicator_slugs:
        indicator_probe_dir = probe_dir / indicator_slug
        if not indicator_probe_dir.exists():
            print(f"Skipping {indicator_slug}: no probe dir found")
            continue

        # Find label_mode subdir (e.g. "span")
        label_mode_dirs = [d for d in indicator_probe_dir.iterdir() if d.is_dir()]
        if not label_mode_dirs:
            print(f"Skipping {indicator_slug}: no label mode dirs")
            continue

        for label_mode_dir in label_mode_dirs:
            label_mode = label_mode_dir.name

            # Discover layer dirs
            layer_dirs = sorted(
                [d for d in label_mode_dir.iterdir() if d.is_dir() and d.name.startswith("layer")],
                key=lambda d: int(d.name.replace("layer", "")),
            )
            if not layer_dirs:
                continue

            # Read indicator name from training_meta.json
            meta_path = layer_dirs[0] / "training_meta.json"
            if not meta_path.exists():
                print(f"Skipping {indicator_slug}/{label_mode}: no training_meta.json")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            indicator_name = meta["indicator_name"]
            detect_layers = [int(d.name.replace("layer", "")) for d in layer_dirs]
            reasoning_only = meta.get("reasoning_only", False)

            print(f"\n{'='*60}")
            print(f"Recomputing thresholds: {indicator_name} ({label_mode})")
            print(f"  Layers: {detect_layers}")
            print(f"{'='*60}")

            # Load training data and split
            data_path = data_dir / f"{indicator_slug}.json"
            if not data_path.exists():
                print(f"  Skipping: {data_path} not found")
                continue
            with open(data_path) as f:
                data = json.load(f)

            _, val_data = _split_transcripts_train_val(data, args.val_fraction)
            val_samples = build_turn_samples(val_data)
            val_n_pos = sum(1 for s in val_samples if s.is_positive)
            val_n_neg = sum(1 for s in val_samples if not s.is_positive)
            print(f"  Val set: {val_n_pos} positive + {val_n_neg} negative turn samples")

            # Extract per-sample val activations
            print("  Extracting val sample activations...")
            val_sample_acts = _extract_val_sample_activations(
                val_samples, model, tokenizer, detect_layers,
                reasoning_only=reasoning_only,
            )
            print(f"  Got {len(val_sample_acts)} val samples with activations")

            # For each layer, load detector and compute thresholds
            for layer_idx, layer_dir in enumerate(layer_dirs):
                layer = detect_layers[layer_idx]
                detector_path = layer_dir / "detector.pt"
                if not detector_path.exists():
                    print(f"  Skipping layer {layer}: no detector.pt")
                    continue

                detector = LogisticRegressionDetector.load(detector_path)
                print(f"\n  Layer {layer}:")

                val_thresholds = _compute_val_thresholds_sentence(
                    detector, val_sample_acts, layer_idx,
                )

                _print_sentence_val_metrics(val_thresholds)

                # Save back into training_meta.json
                meta_path = layer_dir / "training_meta.json"
                with open(meta_path) as f:
                    existing_meta = json.load(f)
                existing_meta["val_thresholds"] = val_thresholds
                with open(meta_path, "w") as f:
                    json.dump(existing_meta, f, indent=2)
                print(f"    Saved updated thresholds to {meta_path}")

            del val_sample_acts
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
