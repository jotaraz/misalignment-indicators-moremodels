"""
Step 3: Build synthetic rollouts for the three ablation conditions.

For each span, creates three rollout entries:
  1. no_context: <think>{span}</think> — span in isolation
  2. positive_context: original misaligned transcript with the span
  3. negative_context: LLM-generated benign transcript with the same span

Output: rollouts saved as rollout.json compatible with probe_eval/evaluate.py

Usage:
    python -m probe_eval.context_ablation.build_rollouts
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _make_event(role: str, content, view=None):
    if view is None:
        view = ["target", "combined"]
    return {
        "type": "transcript_event",
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "edit": {
            "operation": "add",
            "message": {
                "role": role,
                "id": str(uuid.uuid4()),
                "content": content,
            },
        },
    }


def build_no_context_rollout(span: dict, var_num: int) -> dict:
    """Condition 1: span in isolation wrapped in think tags."""
    events = [
        _make_event("system", "You are a helpful assistant."),
        _make_event("user", "Hello."),
        _make_event("assistant", [
            {"type": "reasoning", "reasoning": span["span_text"]},
            {"type": "text", "text": "Hello."},
        ]),
    ]
    return {
        "variation_number": var_num,
        "variation_description": f"[no_context/{span['indicator']}] {span['span_text'][:80]}",
        "repetition_number": 1,
        "modality": "context_ablation",
        "transcript": {
            "metadata": {
                "condition": "no_context",
                "indicator": span["indicator"],
                "span_type": span.get("span_type", "unknown"),
                "span_text": span["span_text"],
                "span_id": span.get("transcript_id", ""),
            },
            "events": events,
        },
    }


def build_positive_context_rollout(span: dict, var_num: int) -> dict:
    """Condition 2: span with its original misaligned context."""
    messages = span.get("messages", [])

    # Strip BloomEvaluator harness if present
    if (messages and messages[0].get("role") == "system"
            and "BloomEvaluator" in messages[0].get("content", "")):
        sys_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
        if len(sys_indices) > 1:
            messages = messages[sys_indices[1]:]

    events = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # For the target turn, wrap content with reasoning
        if role == "assistant" and span["span_text"] in content:
            # Split into reasoning (think) and output
            # The training data often has <think>...</think> format
            think_match = content.find("<think>")
            if think_match >= 0:
                think_end = content.find("</think>")
                if think_end >= 0:
                    reasoning = content[think_match + 7:think_end]
                    output = content[think_end + 8:].strip()
                    events.append(_make_event("assistant", [
                        {"type": "reasoning", "reasoning": reasoning},
                        {"type": "text", "text": output},
                    ]))
                    continue

            # No think tags — put entire content as reasoning
            events.append(_make_event("assistant", [
                {"type": "reasoning", "reasoning": content},
                {"type": "text", "text": ""},
            ]))
        else:
            events.append(_make_event(role, content))

    if not events:
        # Fallback: just the turn text
        events = [
            _make_event("system", "You are a helpful assistant."),
            _make_event("user", "Continue."),
            _make_event("assistant", [
                {"type": "reasoning", "reasoning": span.get("turn_text", span["span_text"])},
                {"type": "text", "text": ""},
            ]),
        ]

    return {
        "variation_number": var_num,
        "variation_description": f"[positive_context/{span['indicator']}] {span['span_text'][:80]}",
        "repetition_number": 1,
        "modality": "context_ablation",
        "transcript": {
            "metadata": {
                "condition": "positive_context",
                "indicator": span["indicator"],
                "span_type": span.get("span_type", "unknown"),
                "span_text": span["span_text"],
                "span_id": span.get("transcript_id", ""),
            },
            "events": events,
        },
    }


def build_negative_context_rollout(span: dict, var_num: int) -> dict:
    """Condition 3: span embedded in LLM-generated benign context."""
    ctx = span.get("benign_context", {})

    events = []
    sys_prompt = ctx.get("system_prompt", "You are a helpful assistant.")
    events.append(_make_event("system", sys_prompt))

    for msg in ctx.get("user_messages", ["Hello."]):
        events.append(_make_event("user", msg))

    think = ctx.get("assistant_think", "")
    resp = ctx.get("assistant_response", "")

    if think:
        events.append(_make_event("assistant", [
            {"type": "reasoning", "reasoning": think},
            {"type": "text", "text": resp},
        ]))
    else:
        events.append(_make_event("assistant", resp))

    return {
        "variation_number": var_num,
        "variation_description": f"[negative_context/{span['indicator']}] {span['span_text'][:80]}",
        "repetition_number": 1,
        "modality": "context_ablation",
        "transcript": {
            "metadata": {
                "condition": "negative_context",
                "indicator": span["indicator"],
                "span_type": span.get("span_type", "unknown"),
                "span_text": span["span_text"],
                "span_id": span.get("transcript_id", ""),
            },
            "events": events,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(DATA_DIR / "spans_with_contexts.json"))
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "rollouts"))
    args = parser.parse_args()

    with open(args.input) as f:
        spans = json.load(f)

    print(f"Loaded {len(spans)} spans with contexts")

    # Filter to spans where benign context has verbatim check passed
    valid = [s for s in spans if s.get("benign_context", {}).get("span_verbatim_check") == "PASSED"]
    print(f"Valid (verbatim pass): {len(valid)}")

    # Build rollouts for all three conditions
    rollouts = []
    span_index = []  # Track which span each rollout corresponds to
    var_num = 1

    for span in valid:
        # Three conditions per span
        r_no = build_no_context_rollout(span, var_num)
        r_pos = build_positive_context_rollout(span, var_num + 1)
        r_neg = build_negative_context_rollout(span, var_num + 2)

        rollouts.extend([r_no, r_pos, r_neg])
        span_index.extend([
            {"span_idx": len(span_index) // 3, "condition": "no_context", "var": var_num},
            {"span_idx": len(span_index) // 3, "condition": "positive_context", "var": var_num + 1},
            {"span_idx": len(span_index) // 3, "condition": "negative_context", "var": var_num + 2},
        ])
        var_num += 3

    # Save rollout
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_data = {
        "metadata": {
            "modality": "context_ablation",
            "evaluator": "n/a",
            "target": "synthetic",
            "dataset": "context_ablation",
            "num_rollouts": len(rollouts),
            "num_spans": len(valid),
            "conditions": ["no_context", "positive_context", "negative_context"],
        },
        "rollouts": rollouts,
    }
    rollout_path = output_dir / "rollout.json"
    with open(rollout_path, "w") as f:
        json.dump(rollout_data, f, indent=2)
    print(f"Saved {len(rollouts)} rollouts ({len(valid)} spans × 3 conditions) -> {rollout_path}")

    # Save span index for later analysis
    index_path = output_dir / "span_index.json"
    with open(index_path, "w") as f:
        json.dump(span_index, f, indent=2)
    print(f"Saved span index -> {index_path}")

    # Also create a dummy GT file (all negative for probe eval)
    gt = {
        "metadata": rollout_data["metadata"],
        "evaluated_rollouts": [
            {
                "variation_number": r["variation_number"],
                "repetition_number": 1,
                "misaligned_turns": [],
                "is_misaligned": False,
            }
            for r in rollouts
        ],
    }
    gt_path = output_dir / "rollout_misalignment_turns.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"Saved dummy GT -> {gt_path}")


if __name__ == "__main__":
    main()
