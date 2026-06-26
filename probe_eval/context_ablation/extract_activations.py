"""
Step 4: Run model forward pass, extract probe scores + raw activations.

Loads the GLM model, runs each rollout through it, and saves:
  - Probe scores per rollout (via existing evaluate.py logic)
  - Raw residual stream activations at probe layers for the target span tokens

The raw activations enable residual stream delta analysis between conditions.

Usage (requires GPU):
    python -m probe_eval.context_ablation.extract_activations
    python -m probe_eval.context_ablation.extract_activations --layers 27 --batch-size 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"

sys.path.insert(0, str(PROJECT_ROOT))


def find_span_token_range(tokenizer, input_ids, span_text: str) -> tuple[int, int] | None:
    """Find the token range that covers the target span text.

    Uses tokenizer.decode() + offset_mapping to avoid BPE artifacts (Ġ) that
    break naive "".join(tokens) matching.
    """
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    span_lower = span_text.lower().strip()

    idx = decoded.lower().find(span_lower)
    if idx < 0:
        # Try prefix match for very long spans
        idx = decoded.lower().find(span_lower[:50])
    if idx < 0:
        return None

    span_end = idx + len(span_text)

    # Re-tokenize with offset_mapping to get char→token mapping
    encoding = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    start_tok = None
    end_tok = None
    for ti, (cs, ce) in enumerate(offsets):
        if start_tok is None and ce > idx:
            start_tok = ti
        if cs >= span_end:
            end_tok = ti
            break
    if end_tok is None:
        end_tok = len(offsets)
    if start_tok is None:
        return None

    return (start_tok, end_tok)


def reconstruct_messages(rollout):
    """Extract messages from rollout events, handling <think> tags."""
    events = rollout["transcript"]["events"]
    messages = []
    for e in events:
        msg = e["edit"]["message"]
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "reasoning":
                    reasoning = block["reasoning"]
                    if reasoning.startswith("<think>"):
                        parts.append(reasoning if reasoning.endswith("</think>") else reasoning + "</think>")
                    else:
                        parts.append(f"<think>{reasoning}</think>")
                elif block.get("type") == "text":
                    parts.append(block["text"])
            content = "".join(parts)
        messages.append({"role": role, "content": content})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str,
                        default=str(DATA_DIR / "rollouts"))
    parser.add_argument("--probe-dir", type=str,
                        default=str(PROJECT_ROOT / "probe/probes/v3_v2_5_combined_v1_span"))
    parser.add_argument("--layers", type=int, nargs="+", default=[27])
    parser.add_argument("--max-batch-tokens", type=int, default=32768,
                        help="Max total tokens per batch (adapts batch size to seq length)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout and span index
    with open(rollout_dir / "rollout.json") as f:
        rollout_data = json.load(f)
    with open(rollout_dir / "span_index.json") as f:
        span_index = json.load(f)

    rollouts = rollout_data["rollouts"]
    print(f"Loaded {len(rollouts)} rollouts, {len(span_index)} span entries")

    # Load model using deception-detection infrastructure
    sys.path.insert(0, str(PROJECT_ROOT / "ood_misalignment_eval/deception-detection"))
    from deception_detection.experiment import ExperimentConfig
    from deception_detection.models import get_model_and_tokenizer

    # Find a probe config to get model name
    probe_dir = Path(args.probe_dir)
    sample_cfg = next(probe_dir.glob("*/span/layer27/cfg.yaml"))
    cfg = ExperimentConfig.from_path(sample_cfg.parent)

    print(f"Loading model: {cfg.model_name}")
    model, tokenizer = get_model_and_tokenizer(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load probe detectors for scoring
    from deception_detection.experiment import Experiment
    detectors = {}
    for layer in args.layers:
        for probe_path in probe_dir.glob(f"*/span/layer{layer}"):
            if not (probe_path / "detector.pt").exists():
                continue
            concept = probe_path.parent.parent.name
            det_cfg = ExperimentConfig.from_path(probe_path)
            det = Experiment(det_cfg).get_detector()
            detectors[(concept, layer)] = det
    print(f"Loaded {len(detectors)} detectors")

    # Pre-cache detector directions on CPU for fast scoring
    det_directions = {}
    det_scalers = {}
    for (concept, layer), det in detectors.items():
        det_directions[(concept, layer)] = det.directions[0].cpu().float()
        if hasattr(det, 'scaler_mean') and det.scaler_mean is not None:
            det_scalers[(concept, layer)] = (det.scaler_mean.cpu(), det.scaler_scale.cpu().clamp(min=1e-8))

    # ================================================================
    # Phase 1: Pre-tokenize all rollouts and find span ranges
    # ================================================================
    print("Phase 1: Pre-tokenizing and finding span ranges...")
    tokenized = []  # (ri, input_ids_list, span_range, metadata)

    for ri, rollout in enumerate(rollouts):
        span_text = rollout["transcript"]["metadata"].get("span_text", "")
        messages = reconstruct_messages(rollout)

        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            clear_thinking=False,
        )
        encoding = tokenizer(chat_text, truncation=True, max_length=4096)
        ids = encoding["input_ids"]

        span_range = find_span_token_range(tokenizer, ids, span_text)

        tokenized.append({
            "ri": ri,
            "input_ids": ids,
            "seq_len": len(ids),
            "span_range": span_range,
            "span_text": span_text,
            "condition": rollout["transcript"]["metadata"].get("condition", ""),
            "indicator": rollout["transcript"]["metadata"].get("indicator", ""),
            "variation_number": rollout["variation_number"],
        })

    n_with_span = sum(1 for t in tokenized if t["span_range"] is not None)
    print(f"  Span found: {n_with_span}/{len(tokenized)}")

    # ================================================================
    # Phase 2: Sort by length and process in batches
    # ================================================================
    print(f"Phase 2: Batched forward passes (max_batch_tokens={args.max_batch_tokens})...")

    # Sort by sequence length for efficient padding
    tokenized.sort(key=lambda t: t["seq_len"])

    # Register hooks once
    layer_activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            layer_activations[layer_idx] = hidden.detach().cpu()
        return hook_fn

    hooks = []
    for layer_idx in args.layers:
        layer_module = model.model.layers[layer_idx]
        h = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Process in dynamic batches (group by similar length, cap total tokens)
    results = [None] * len(rollouts)  # indexed by ri
    n_processed = 0

    idx = 0
    while idx < len(tokenized):
        # Build batch up to max_batch_tokens
        batch = []
        batch_tokens = 0
        while idx < len(tokenized):
            seq_len = tokenized[idx]["seq_len"]
            # Would this item push us over? (use max_len_in_batch * new_size as estimate)
            new_max = max(seq_len, batch[-1]["seq_len"] if batch else 0)
            if batch and new_max * (len(batch) + 1) > args.max_batch_tokens:
                break
            batch.append(tokenized[idx])
            idx += 1
        if not batch:
            # Single sequence too long, process alone
            batch = [tokenized[idx]]
            idx += 1

        # Pad batch
        max_len = max(t["seq_len"] for t in batch)
        padded_ids = []
        attention_masks = []
        for t in batch:
            ids = t["input_ids"]
            pad_len = max_len - len(ids)
            # Left-pad (standard for causal LMs)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=model.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=model.device)

        # Forward pass
        layer_activations.clear()
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)

        # Extract per-sample results
        for bi, t in enumerate(batch):
            ri = t["ri"]
            span_range = t["span_range"]
            pad_offset = max_len - t["seq_len"]

            probe_scores = {}
            span_activations = {}

            for layer_idx in args.layers:
                if layer_idx not in layer_activations:
                    continue

                acts = layer_activations[layer_idx][bi]  # [max_len, hidden_dim]

                if span_range:
                    start, end = span_range
                    # Adjust for left-padding offset
                    start_padded = start + pad_offset
                    end_padded = end + pad_offset
                    span_acts = acts[start_padded:end_padded]  # [span_len, hidden_dim]
                    span_activations[f"layer{layer_idx}"] = span_acts.float().numpy()

                    # Score with each probe
                    scored_acts = span_acts.float()
                    for (concept, l), direction in det_directions.items():
                        if l != layer_idx:
                            continue
                        sa = scored_acts
                        if (concept, l) in det_scalers:
                            mean, scale = det_scalers[(concept, l)]
                            sa = (sa - mean) / scale
                        token_scores = (sa @ direction).numpy()
                        probe_scores[(concept, layer_idx)] = {
                            "mean": float(np.mean(token_scores)),
                            "max": float(np.max(token_scores)),
                            "scores": token_scores.tolist(),
                        }

            result = {
                "rollout_index": ri,
                "variation_number": t["variation_number"],
                "condition": t["condition"],
                "indicator": t["indicator"],
                "span_text": t["span_text"],
                "span_range": list(span_range) if span_range else None,
                "n_tokens": t["seq_len"],
                "probe_scores": {
                    f"{concept}__layer{layer}": scores
                    for (concept, layer), scores in probe_scores.items()
                },
            }
            results[ri] = result

            # Save raw activations
            if span_activations:
                act_path = output_dir / f"activations_{ri:04d}.npz"
                np.savez_compressed(act_path, **span_activations)

        n_processed += len(batch)
        if n_processed % 200 == 0 or n_processed == len(tokenized):
            print(f"  Processed {n_processed}/{len(tokenized)} (batch={len(batch)}, max_len={max(t['seq_len'] for t in batch)})", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save probe scores
    scores_path = output_dir / "probe_scores.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe scores -> {scores_path}")
    print(f"Saved activations -> {output_dir}/activations_*.npz")


if __name__ == "__main__":
    main()
