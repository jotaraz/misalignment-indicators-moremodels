"""
Step 6: Generate matched training/val examples for context-dependent detection.

From the context ablation data, creates probe training examples where:
  - Positive: span in its original misaligned context
  - Negative: same span in a benign context

This tests whether a linear probe can learn to distinguish context-dependent
misalignment from benign usage of the same text.

Produces two datasets:
  1. context_only: Only context-dependent pairs (tests if model can learn
     context-dependent features at all)
  2. combined: Context-dependent pairs + original training data (tests if
     one probe can handle both context-dependent and context-independent cases)

Usage:
    python -m probe_eval.context_ablation.generate_training_data
    python -m probe_eval.context_ablation.generate_training_data --val-fraction 0.2
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
TRAINING_DATA_DIR = PROJECT_ROOT / "probe/data/v3_v2_5_combined_v1"


def load_original_training(indicator: str) -> list[dict]:
    """Load original training transcripts for an indicator."""
    path = TRAINING_DATA_DIR / f"{indicator}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("transcripts", [])


def fix_alternating_roles(messages: list[dict]) -> list[dict]:
    """Insert separator messages to maintain alternating user/assistant roles.

    Same technique as evaluate.py's extract_target_dialogue — when two consecutive
    messages have the same role (e.g., user→user after tool→user mapping), insert
    a dummy separator to satisfy chat template requirements.
    """
    if not messages:
        return messages
    fixed = [messages[0]]
    for m in messages[1:]:
        if fixed[-1]["role"] == m["role"]:
            sep_role = "assistant" if m["role"] == "user" else "user"
            fixed.append({"role": sep_role, "content": "(continued)"})
        fixed.append(m)
    return fixed


def build_context_dependent_examples(spans_with_contexts: list[dict]) -> dict[str, list[dict]]:
    """Build per-indicator training examples from context ablation pairs.

    Each example is a transcript with per_turn_labels marking the span turn.
    Positive = original misaligned context, Negative = benign context with same span.
    """
    by_indicator = {}

    for span in spans_with_contexts:
        if span.get("benign_context", {}).get("span_verbatim_check") != "PASSED":
            continue

        indicator = span["indicator"]
        # Dev set spans have indicator="_dev_set" — group them separately
        # They'll be added to the combined dataset across all indicators
        if indicator not in by_indicator:
            by_indicator[indicator] = []

        span_text = span["span_text"]

        # Positive example: original misaligned context
        pos_messages = span.get("messages", [])
        if pos_messages:
            # Strip BloomEvaluator harness: if first system msg contains
            # "BloomEvaluator", skip to the second system msg (inner conversation)
            if (pos_messages[0].get("role") == "system"
                    and "BloomEvaluator" in pos_messages[0].get("content", "")):
                sys_indices = [i for i, m in enumerate(pos_messages) if m["role"] == "system"]
                if len(sys_indices) > 1:
                    pos_messages = pos_messages[sys_indices[1]:]

            # Fix alternating roles (tool→user mapping creates consecutive user msgs)
            pos_messages = fix_alternating_roles(pos_messages)

            # Find which turn contains the span
            span_turn = None
            for i, msg in enumerate(pos_messages):
                if span_text in msg.get("content", ""):
                    span_turn = i
                    break

            pos_example = {
                "messages": pos_messages,
                "per_turn_labels": [],
                "label": "deceptive",
                "id": f"ctx_dep_pos_{indicator}_{len(by_indicator[indicator])}",
                "negative_type": None,
                "context_dependent": True,
                "span_text": span_text,
                "source": span.get("source", "training"),
            }
            if span_turn is not None:
                pos_example["per_turn_labels"].append({
                    "turn_index": span_turn,
                    "has_indicator": True,
                    "spans": [span_text],
                })
            by_indicator[indicator].append(pos_example)

        # Negative example: benign context with same span
        ctx = span["benign_context"]
        neg_messages = []
        if ctx.get("system_prompt"):
            neg_messages.append({"role": "system", "content": ctx["system_prompt"]})
        for um in ctx.get("user_messages", []):
            neg_messages.append({"role": "user", "content": um})

        # Build assistant message with think tags if present
        think = ctx.get("assistant_think", "")
        resp = ctx.get("assistant_response", "")
        if think:
            asst_content = f"<think>{think}</think>{resp}"
        else:
            asst_content = resp
        neg_messages.append({"role": "assistant", "content": asst_content})

        # Find which turn contains the span in negative context
        neg_span_turn = None
        for i, msg in enumerate(neg_messages):
            if span_text in msg.get("content", ""):
                neg_span_turn = i
                break

        neg_example = {
            "messages": neg_messages,
            "per_turn_labels": [],
            "label": "honest",
            "id": f"ctx_dep_neg_{indicator}_{len(by_indicator[indicator])}",
            "negative_type": "context_dependent_benign",
            "context_dependent": True,
            "span_text": span_text,
        }
        if neg_span_turn is not None:
            neg_example["per_turn_labels"].append({
                "turn_index": neg_span_turn,
                "has_indicator": False,
                "spans": [],
            })
        by_indicator[indicator].append(neg_example)

    return by_indicator


def split_train_val(examples: list[dict], val_fraction: float, seed: int = 42) -> tuple[list, list]:
    """Split examples into train/val, keeping positive/negative pairs together."""
    # Group by span_text to keep paired examples together
    by_span = {}
    for ex in examples:
        span = ex.get("span_text", ex.get("id", ""))
        if span not in by_span:
            by_span[span] = []
        by_span[span].append(ex)

    span_keys = list(by_span.keys())
    random.seed(seed)
    random.shuffle(span_keys)

    n_val = max(1, int(len(span_keys) * val_fraction))
    val_keys = set(span_keys[:n_val])

    train = []
    val = []
    for key in span_keys:
        if key in val_keys:
            val.extend(by_span[key])
        else:
            train.extend(by_span[key])

    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "training"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load context ablation data
    spans_path = DATA_DIR / "spans_with_contexts.json"
    with open(spans_path) as f:
        spans_with_contexts = json.load(f)
    print(f"Loaded {len(spans_with_contexts)} spans with contexts")

    # Build context-dependent examples
    ctx_examples = build_context_dependent_examples(spans_with_contexts)
    total_ctx = sum(len(v) for v in ctx_examples.values())
    print(f"Built {total_ctx} context-dependent examples across {len(ctx_examples)} indicators")

    # ================================================================
    # Dataset 1: context_only — only context-dependent pairs
    # Format compatible with probe/train.py: {"transcripts": [...]}
    # ================================================================
    ctx_only_dir = output_dir / "context_only"
    ctx_only_dir.mkdir(parents=True, exist_ok=True)

    for indicator, examples in ctx_examples.items():
        # Filter out examples with empty per_turn_labels for deceptive
        valid = [e for e in examples
                 if e["label"] != "deceptive" or e.get("per_turn_labels")]
        skipped = len(examples) - len(valid)

        n_pos = sum(1 for e in valid if e["label"] == "deceptive")
        n_neg = sum(1 for e in valid if e["label"] != "deceptive")

        # Save in probe/train.py format: {"transcripts": [...]}
        data = {
            "indicator_name": indicator,
            "indicator_definition": "",
            "indicator_category": "context_dependent",
            "generation_model": "context_ablation",
            "transcripts": valid,
        }
        path = ctx_only_dir / f"{indicator}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if skipped:
            print(f"  context_only/{indicator}: {n_pos}+/{n_neg}- (skipped {skipped} with no span match)")
        else:
            print(f"  context_only/{indicator}: {n_pos}+/{n_neg}-")

    # ================================================================
    # Dataset 2: combined — original training + dev positives + all ctx negatives
    #   - Original training data (positives + negatives) — as-is
    #   - Dev-sourced positives (new bloom rollout transcripts not in original training)
    #   - All context_only negatives (benign convos with matched span text)
    #   - Skips training-sourced positives (already covered by original data)
    # ================================================================
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    for indicator, ctx_exs in ctx_examples.items():
        # Load original training data
        orig_transcripts = load_original_training(indicator)

        if not orig_transcripts:
            print(f"  SKIP combined/{indicator}: no original training data")
            continue

        # Filter ctx examples: keep dev-sourced positives + ALL negatives
        # Skip training-sourced positives (redundant with original data)
        valid_ctx = []
        n_dev_pos = 0
        n_ctx_neg = 0
        n_skipped_dup = 0
        for e in ctx_exs:
            if e["label"] == "deceptive":
                # Only keep dev-sourced positives (not training-sourced duplicates)
                source = e.get("source", "")
                if source and source.startswith("dev_"):
                    if e.get("per_turn_labels"):  # must have valid span labels
                        valid_ctx.append(e)
                        n_dev_pos += 1
                else:
                    n_skipped_dup += 1
            else:
                # Keep ALL negatives (benign conversations with matched span text)
                valid_ctx.append(e)
                n_ctx_neg += 1

        # Combine
        all_transcripts = orig_transcripts + valid_ctx

        random.seed(42)
        random.shuffle(all_transcripts)

        n_pos = sum(1 for e in all_transcripts if e.get("label") == "deceptive")
        n_neg = len(all_transcripts) - n_pos
        n_ctx = sum(1 for e in all_transcripts if e.get("context_dependent"))

        # Save in probe/train.py format
        data = {
            "indicator_name": indicator,
            "indicator_definition": "",
            "indicator_category": "combined",
            "generation_model": "context_ablation_combined",
            "transcripts": all_transcripts,
        }
        path = combined_dir / f"{indicator}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n_orig = len(orig_transcripts)
        print(f"  combined/{indicator}: {n_pos}+/{n_neg}- "
              f"(orig={n_orig}, +{n_dev_pos} dev pos, +{n_ctx_neg} ctx neg, skipped {n_skipped_dup} dup pos)")

    # ================================================================
    # Dataset 3: combined_sentence — original training + sentence-level ctx pairs only
    #   - Original training data (positives + negatives) — as-is
    #   - Dev-sourced sentence positives only
    #   - Sentence-level context negatives only (not short phrase negatives)
    #   - Short phrase pairs excluded (probes already handle those well)
    # ================================================================
    combined_sent_dir = output_dir / "combined_sentence"
    combined_sent_dir.mkdir(parents=True, exist_ok=True)

    # Load span types from spans_with_contexts
    span_type_lookup = {}
    for span in spans_with_contexts:
        key = (span["indicator"], span["span_text"])
        span_type_lookup[key] = span.get("span_type", "unknown")

    for indicator, ctx_exs in ctx_examples.items():
        orig_transcripts = load_original_training(indicator)
        if not orig_transcripts:
            print(f"  SKIP combined_sentence/{indicator}: no original training data")
            continue

        valid_ctx = []
        n_dev_pos = 0
        n_ctx_neg = 0
        n_skipped_short = 0
        n_skipped_dup = 0
        for e in ctx_exs:
            span_text = e.get("span_text", "")
            stype = span_type_lookup.get((indicator, span_text), "unknown")

            # Only keep sentence-level spans
            if stype != "sentence":
                n_skipped_short += 1
                continue

            if e["label"] == "deceptive":
                source = e.get("source", "")
                if source and source.startswith("dev_"):
                    if e.get("per_turn_labels"):
                        valid_ctx.append(e)
                        n_dev_pos += 1
                else:
                    n_skipped_dup += 1
            else:
                valid_ctx.append(e)
                n_ctx_neg += 1

        all_transcripts = orig_transcripts + valid_ctx
        random.seed(42)
        random.shuffle(all_transcripts)

        n_pos = sum(1 for e in all_transcripts if e.get("label") == "deceptive")
        n_neg = len(all_transcripts) - n_pos

        data = {
            "indicator_name": indicator,
            "indicator_definition": "",
            "indicator_category": "combined_sentence",
            "generation_model": "context_ablation_combined_sentence",
            "transcripts": all_transcripts,
        }
        path = combined_sent_dir / f"{indicator}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n_orig = len(orig_transcripts)
        print(f"  combined_sentence/{indicator}: {n_pos}+/{n_neg}- "
              f"(orig={n_orig}, +{n_dev_pos} dev pos, +{n_ctx_neg} sent neg, skipped {n_skipped_short} short)")

    print(f"\nSaved datasets to {output_dir}/")
    print(f"  context_only/: {len(ctx_examples)} indicators (context-dependent pairs only)")
    print(f"  combined/: context-dependent + original training data")
    print(f"  combined_sentence/: original + sentence-level ctx pairs only")


if __name__ == "__main__":
    main()
