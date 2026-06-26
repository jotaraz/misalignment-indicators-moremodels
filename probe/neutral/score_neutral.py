"""
Score neutral conversations with all probes and find false-positive sentences.

For each probe (layer 27), loads the clean_label threshold_f1, scores every
neutral dialogue at the sentence level, and records sentences that exceed the
threshold.

Usage:
    python -m probe.neutral.score_neutral
    python -m probe.neutral.score_neutral --max-dialogues 500
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

REPO_ROOT = Path(__file__).parent.parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.activations import Activations
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue, Message

MODEL_NAME = ModelName.GLM_FLASH
DEFAULT_LAYER = 27
DEFAULT_PADDING = {
    "gemma": {"left": 0, "right": 0},
    "mistral": {"left": 0, "right": 0},
    "llama": {"left": 0, "right": 0},
    "qwen": {"left": 0, "right": 0},
    "glm": {"left": 0, "right": 0},
}
DEFAULT_PROBES_DIR = REPO_ROOT / "probe" / "probes" / "v2_3_gen_prompt_v2_span_v2"
DEFAULT_DIALOGUES_PATH = REPO_ROOT / "probe" / "data" / "neutral" / "dialogues_filtered_v3.json"

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _split_tokens_into_sentences(
    str_tokens: list[str],
    scores: list[float],
) -> list[tuple[float, str]]:
    """Split scored tokens into sentences, returning (mean_score, sentence_text) pairs."""
    if not str_tokens:
        return []

    text = "".join(str_tokens)
    split_positions = sorted(set(m.end() for m in _SENTENCE_SPLIT_RE.finditer(text)))

    sentence_scores: list[list[float]] = [[]]
    sentence_tokens: list[list[str]] = [[]]
    char_pos = 0
    split_idx = 0

    for i, tok in enumerate(str_tokens):
        tok_mid = char_pos + len(tok) / 2
        while split_idx < len(split_positions) and tok_mid >= split_positions[split_idx]:
            split_idx += 1
            sentence_scores.append([])
            sentence_tokens.append([])
        sentence_scores[-1].append(scores[i])
        sentence_tokens[-1].append(tok)
        char_pos += len(tok)

    # Merge short sentences (< 5 chars stripped) into adjacent ones
    i = 0
    while i < len(sentence_scores):
        sent_text = "".join(sentence_tokens[i]).strip()
        if len(sent_text) < 5:
            if i > 0:
                sentence_scores[i - 1].extend(sentence_scores[i])
                sentence_tokens[i - 1].extend(sentence_tokens[i])
                sentence_scores.pop(i)
                sentence_tokens.pop(i)
            elif i < len(sentence_scores) - 1:
                sentence_scores[i + 1] = sentence_scores[i] + sentence_scores[i + 1]
                sentence_tokens[i + 1] = sentence_tokens[i] + sentence_tokens[i + 1]
                sentence_scores.pop(i)
                sentence_tokens.pop(i)
            else:
                i += 1
        else:
            i += 1

    result: list[tuple[float, str]] = []
    for s_scores, s_tokens in zip(sentence_scores, sentence_tokens):
        if s_scores:
            result.append((float(np.mean(s_scores)), "".join(s_tokens)))
    return result


def load_probes(
    probes_dir: Path,
    layer: int,
) -> dict[str, tuple[LogisticRegressionDetector, float, str]]:
    """Load all probe detectors and their clean_label threshold_f1 for a given layer.

    Returns dict mapping indicator_slug -> (detector, threshold, indicator_name).
    """
    probes: dict[str, tuple[LogisticRegressionDetector, float, str]] = {}

    for indicator_dir in sorted(probes_dir.iterdir()):
        if not indicator_dir.is_dir():
            continue
        detector_path = indicator_dir / "span" / f"layer{layer}" / "detector.pt"
        meta_path = indicator_dir / "span" / f"layer{layer}" / "training_meta.json"
        if not detector_path.exists() or not meta_path.exists():
            continue

        detector = LogisticRegressionDetector.load(detector_path)

        with open(meta_path) as f:
            meta = json.load(f)

        clean_label = meta.get("val_thresholds", {}).get("clean_label", {})
        threshold = clean_label.get("threshold_f1")
        if threshold is None:
            print(f"  WARNING: no clean_label threshold_f1 for {indicator_dir.name}, skipping")
            continue

        indicator_name = meta.get("indicator_name", indicator_dir.name)
        probes[indicator_dir.name] = (detector, float(threshold), indicator_name)
        print(f"  Loaded {indicator_name}: threshold_f1={threshold:.4f}")

    return probes


def _build_length_batches(
    dialogues: list[dict],
    tokenizer: Any,
    max_tokens_per_batch: int = 12000,
) -> list[list[int]]:
    """Group dialogue indices into batches with similar token lengths.

    Sorts by estimated token count, then greedily fills batches up to
    max_tokens_per_batch total tokens (accounting for padding to max in batch).
    """
    # Estimate token counts (rough: 4 chars per token)
    indexed = [(i, len(d["assistant_response"]) // 4 + 100) for i, d in enumerate(dialogues)]
    indexed.sort(key=lambda x: x[1])

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_len = 0

    for idx, est_tokens in indexed:
        new_max = max(current_max_len, est_tokens)
        new_total = new_max * (len(current_batch) + 1)
        if current_batch and new_total > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_max_len = est_tokens
        else:
            current_batch.append(idx)
            current_max_len = new_max

    if current_batch:
        batches.append(current_batch)

    return batches


def score_dialogues(
    dialogues: list[dict],
    probes: dict[str, tuple[LogisticRegressionDetector, float, str]],
    model: Any,
    tokenizer: Any,
    layer: int,
) -> list[dict]:
    """Score all dialogues and return per-sentence per-probe scores.

    Groups dialogues by similar token length and batches the forward pass
    for much better GPU utilization than one-at-a-time processing.

    Returns a list of result dicts, one per scored dialogue, each with
    ``dialogue_index``, prompts, response, the list of sentence texts, and
    ``probe_scores`` mapping probe slug to a list of per-sentence mean
    scores. Threshold filtering is done downstream so different thresholds
    can be applied without re-running the forward pass.
    """
    import sys
    template_kwargs: dict[str, Any] = {"clear_thinking": False}
    results: list[dict] = []

    batches = _build_length_batches(dialogues, tokenizer)
    total_dialogues = sum(len(b) for b in batches)
    print(f"  {total_dialogues} dialogues grouped into {len(batches)} batches "
          f"(sizes: {min(len(b) for b in batches)}-{max(len(b) for b in batches)})",
          flush=True)

    processed = 0
    for batch_i, batch_indices in enumerate(batches):
        # Tokenize all dialogues in this batch together
        dialogue_lists: list[Dialogue] = []
        for idx in batch_indices:
            d = dialogues[idx]
            dialogue_lists.append([
                Message(role="system", content=d["system_prompt"], detect=False),
                Message(role="user", content=d["user_prompt"], detect=False),
                Message(role="assistant", content=d["assistant_response"], detect=True),
            ])

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=dialogue_lists,
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                template_kwargs=template_kwargs,
            )
        except Exception as e:
            print(f"  Batch {batch_i} tokenization failed: {e}", flush=True)
            processed += len(batch_indices)
            continue

        # Batched forward pass
        try:
            acts = Activations.from_model(
                model, toks, batch_size=len(batch_indices), layers=[layer],
            )
        except Exception as e:
            print(f"  Batch {batch_i} activation extraction failed: {e}", flush=True)
            processed += len(batch_indices)
            continue

        # Score each dialogue in the batch
        for pos, orig_idx in enumerate(batch_indices):
            d = dialogues[orig_idx]

            if toks.detection_mask is not None and not toks.detection_mask[pos].any():
                continue

            mask = toks.detection_mask[pos]
            detected_indices = mask.nonzero(as_tuple=True)[0]
            if len(detected_indices) == 0:
                continue

            detected_str_tokens = [toks.str_tokens[pos][j.item()] for j in detected_indices]

            # all_acts shape: [batch, seqpos, layer, emb]
            masked = acts.all_acts[pos][mask]  # [n_detected, layer, emb]
            if masked.numel() == 0:
                continue

            sentence_texts: list[str] | None = None
            probe_sentence_scores: dict[str, list[float]] = {}
            for slug, (detector, threshold, indicator_name) in probes.items():
                token_scores = detector.get_score_tensor(masked).detach().cpu().numpy()
                sentences = _split_tokens_into_sentences(
                    detected_str_tokens, token_scores.tolist()
                )
                if sentence_texts is None:
                    sentence_texts = [t.strip() for _, t in sentences]
                probe_sentence_scores[slug] = [round(float(s), 4) for s, _ in sentences]

            if sentence_texts is None or not probe_sentence_scores:
                continue

            results.append({
                "dialogue_index": orig_idx,
                "system_prompt": d["system_prompt"],
                "user_prompt": d["user_prompt"],
                "assistant_response": d["assistant_response"],
                "sentences": sentence_texts,
                "probe_scores": probe_sentence_scores,
            })

        del acts, toks
        torch.cuda.empty_cache()
        processed += len(batch_indices)
        print(f"  Batch {batch_i+1}/{len(batches)}: {processed}/{total_dialogues} done "
              f"({len(results)} scored)", flush=True)
        sys.stdout.flush()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score neutral dialogues with probes and find false positives"
    )
    parser.add_argument(
        "--probes-dir", type=str, default=None,
        help=f"Probes directory (default: {DEFAULT_PROBES_DIR})",
    )
    parser.add_argument(
        "--dialogues-path", type=str, default=None,
        help=f"Path to neutral dialogues JSON (default: {DEFAULT_DIALOGUES_PATH})",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help=f"Layer to score (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Start index into dialogues list (default: 0)",
    )
    parser.add_argument(
        "--max-dialogues", type=int, default=None,
        help="Limit number of dialogues to score",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: probe/data/neutral/false_positives.json)",
    )
    parser.add_argument(
        "--thresholds-file", type=str, default=None,
        help="JSON file with custom thresholds (overrides clean_label threshold_f1)",
    )
    parser.add_argument(
        "--thresholds-version", type=str, default=None,
        help="Version key inside thresholds file (e.g. 'fpr_0.02')",
    )
    args = parser.parse_args()

    probes_dir = Path(args.probes_dir) if args.probes_dir else DEFAULT_PROBES_DIR
    dialogues_path = Path(args.dialogues_path) if args.dialogues_path else DEFAULT_DIALOGUES_PATH
    output_path = Path(args.output) if args.output else (
        REPO_ROOT / "probe" / "data" / "neutral" / "false_positives.json"
    )

    # Load dialogues
    print(f"Loading dialogues from {dialogues_path}...")
    with open(dialogues_path) as f:
        all_dialogues = json.load(f)
    dialogues = all_dialogues[args.offset:]
    if args.max_dialogues:
        dialogues = dialogues[:args.max_dialogues]
    print(f"  {len(dialogues)} dialogues to score (offset={args.offset})")

    # Load probes
    print(f"\nLoading probes from {probes_dir} (layer {args.layer})...")
    probes = load_probes(probes_dir, args.layer)
    print(f"  Loaded {len(probes)} probes")

    # Override thresholds if custom file provided
    if args.thresholds_file:
        with open(args.thresholds_file) as f:
            thresh_data = json.load(f)
        version = args.thresholds_version or next(iter(thresh_data["versions"]))
        layer_key = f"layer{args.layer}"
        custom = thresh_data["versions"][version]["per_layer"][layer_key]["thresholds"]
        print(f"\n  Overriding thresholds from {args.thresholds_file} (version={version}):")
        for slug in list(probes.keys()):
            if slug in custom:
                detector, _, indicator_name = probes[slug]
                probes[slug] = (detector, float(custom[slug]), indicator_name)
                print(f"    {indicator_name}: {custom[slug]:.4f}")
            else:
                print(f"    WARNING: no custom threshold for {slug}, keeping default")

    # Load model
    print(f"\nLoading model {MODEL_NAME.value}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    # Warmup
    print("Running warmup pass...")
    try:
        d = dialogues[0]
        warmup_dialogue: Dialogue = [
            Message(role="system", content=d["system_prompt"], detect=False),
            Message(role="user", content=d["user_prompt"], detect=False),
            Message(role="assistant", content=d["assistant_response"], detect=True),
        ]
        warmup_toks = TokenizedDataset.from_dialogue_list(
            dialogues=[warmup_dialogue],
            tokenizer=tokenizer,
            padding=DEFAULT_PADDING,
            detect_all=True,
            template_kwargs={"clear_thinking": False},
        )
        Activations.from_model(model, warmup_toks, batch_size=1, layers=[args.layer])
    except Exception:
        pass

    # Score
    print(f"\nScoring {len(dialogues)} dialogues...")
    results = score_dialogues(dialogues, probes, model, tokenizer, args.layer)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # Derive triggered_dialogues at the default thresholds for backward compat
    triggered_dialogues: list[dict] = []
    probe_counts: dict[str, int] = {}
    for r in results:
        triggered_probes: dict[str, dict] = {}
        for slug, scores in r["probe_scores"].items():
            _, threshold, indicator_name = probes[slug]
            exceeding = [
                {
                    "sentence": r["sentences"][i],
                    "score": s,
                    "threshold": round(float(threshold), 4),
                }
                for i, s in enumerate(scores) if s > threshold
            ]
            if exceeding:
                triggered_probes[slug] = {
                    "indicator_name": indicator_name,
                    "threshold": round(float(threshold), 4),
                    "sentences": exceeding,
                }
                probe_counts[slug] = probe_counts.get(slug, 0) + 1
        if triggered_probes:
            triggered_dialogues.append({
                "dialogue_index": r["dialogue_index"],
                "system_prompt": r["system_prompt"],
                "user_prompt": r["user_prompt"],
                "assistant_response": r["assistant_response"],
                "triggered_probes": triggered_probes,
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"Scored: {len(results)} / {len(dialogues)} dialogues")
    print(f"Triggered (at default thresholds): {len(triggered_dialogues)}")

    print(f"\nPer-probe trigger counts:")
    for slug, count in sorted(probe_counts.items(), key=lambda x: -x[1]):
        _, threshold, name = probes[slug]
        print(f"  {name}: {count} dialogues (threshold={threshold:.4f})")

    # Save
    output = {
        "n_dialogues_scored": len(dialogues),
        "n_dialogues_triggered": len(triggered_dialogues),
        "layer": args.layer,
        "probes_dir": str(probes_dir),
        "probes": {
            slug: {
                "indicator_name": probes[slug][2],
                "default_threshold": round(probes[slug][1], 4),
            }
            for slug in probes
        },
        "per_probe_counts": {
            slug: {
                "indicator_name": probes[slug][2],
                "threshold": round(probes[slug][1], 4),
                "n_triggered": count,
            }
            for slug, count in sorted(probe_counts.items(), key=lambda x: -x[1])
        },
        "scored_dialogues": results,
        "triggered_dialogues": triggered_dialogues,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
