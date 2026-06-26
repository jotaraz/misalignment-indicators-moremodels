#!/usr/bin/env python3
"""Download LoRA weights from Tinker for all neutral seeds.

Usage:
    export TINKER_API_KEY=<your key>
    python download_tinker_lora.py [--output-dir OUTPUT_DIR] [--seeds 0 1 2]

Downloads LoRA adapter archives (.tar) for each seed checkpoint.
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    import tinker
except ImportError:
    print("Error: tinker SDK not installed. Install with: pip install tinker")
    sys.exit(1)


# Checkpoint paths for all 10 neutral seeds (Qwen3-30B-A3B-Base, 4k, neutral)
SEED_CHECKPOINTS = {
    0: "tinker://8a08de51-589c-59f5-9244-654028e5eeb0:train:0/sampler_weights/final",
    1: "tinker://10f9a922-42b4-552b-a0be-f7320b5b14fa:train:0/sampler_weights/final",
    2: "tinker://ca886717-b5a3-5f34-a2fe-24736a9398cf:train:0/sampler_weights/final",
    3: "tinker://b4898733-0624-53e0-8e9c-96c1d0bc363e:train:0/sampler_weights/final",
    4: "tinker://cee288c3-da4c-5445-9d8d-2cd00d100613:train:0/sampler_weights/final",
    5: "tinker://44db35f8-6e37-5bdf-8dd4-6392d04b6700:train:0/sampler_weights/final",
    6: "tinker://ae3e850d-cf4f-5d00-94ac-0229867bbc0a:train:0/sampler_weights/final",
    7: "tinker://498d84e7-7c01-5aae-8f03-6e7f8e035fb7:train:0/sampler_weights/final",
    8: "tinker://a6ee27fd-9870-56a2-b511-6493d874e312:train:0/sampler_weights/final",
    9: "tinker://ade28b5a-d5ac-5c26-a327-3920e77c3dd1:train:0/sampler_weights/final",
}


def download_checkpoint(tinker_path: str, output_dir: Path, seed: int) -> Path:
    """Download a single checkpoint's LoRA weights."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    archive_path = seed_dir / "archive.tar"
    if archive_path.exists():
        print(f"  [skip] {archive_path} already exists")
        return seed_dir

    print(f"  Requesting download URL for: {tinker_path}")
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    response = future.result()

    print(f"  Downloading archive (URL expires: {response.expires})...")
    urllib.request.urlretrieve(response.url, str(archive_path))
    print(f"  Saved to {archive_path}")

    # Extract the archive
    print(f"  Extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=seed_dir)
    print(f"  Extracted to {seed_dir}")

    return seed_dir


def main():
    parser = argparse.ArgumentParser(description="Download Tinker LoRA weights for neutral seeds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tinker-lora-weights/neutral"),
        help="Output directory for downloaded weights (default: tinker-lora-weights/neutral)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seeds to download (default: all 0-9)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract tar archives after download",
    )
    args = parser.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        print("Error: TINKER_API_KEY environment variable not set")
        sys.exit(1)

    seeds = args.seeds if args.seeds is not None else list(range(10))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading LoRA weights for {len(seeds)} seeds to {args.output_dir}/")
    print()

    for seed in seeds:
        if seed not in SEED_CHECKPOINTS:
            print(f"[warn] Seed {seed} not in checkpoint list, skipping")
            continue

        print(f"[seed_{seed}]")
        try:
            download_checkpoint(SEED_CHECKPOINTS[seed], args.output_dir, seed)
        except Exception as e:
            print(f"  [ERROR] Failed to download seed {seed}: {e}")
        print()

    print("Done!")
    print(f"Weights saved to: {args.output_dir}/")
    print()
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    for seed in seeds:
        print(f"    seed_{seed}/")
        print(f"      archive.tar")
        print(f"      (extracted LoRA adapter files)")


if __name__ == "__main__":
    main()
