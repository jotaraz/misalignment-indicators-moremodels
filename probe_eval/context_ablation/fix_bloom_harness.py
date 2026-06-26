"""
Fix BloomEvaluator harness contamination in context ablation data files.

Some dev-sourced spans include evaluator-view messages (BloomEvaluator system prompt,
<target_thinking> wrappers, evaluator assistant turns) mixed into the target model's
conversation. This script reloads the correct target-view-only messages from the
original bloom rollout files and patches all affected data.

Affected files:
  - spans_with_contexts.json (messages field)
  - rollouts/rollout.json (positive_context events)
  - training/context_only/*.json (transcripts_train/val messages)
  - training/combined/*.json (transcripts_train/val messages)

Usage:
    python -m probe_eval.context_ablation.fix_bloom_harness
    python -m probe_eval.context_ablation.fix_bloom_harness --dry-run
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
BLOOM_RESULTS_DIR = PROJECT_ROOT / "bloom/bloom-results"

ROLLOUT_SEARCH_PATHS = [
    BLOOM_RESULTS_DIR,
    PROJECT_ROOT / "bloom/bloom-results-test",
]


def load_target_messages(behavior: str, variation: int) -> list[dict]:
    """Load target-view-only messages from a bloom rollout, filtering by view field."""
    for base in ROLLOUT_SEARCH_PATHS:
        rpath = base / behavior / "rollout.json"
        if rpath.exists():
            break
    else:
        return []

    with open(rpath) as f:
        data = json.load(f)

    for r in data.get("rollouts", []):
        if r.get("variation_number") != variation:
            continue
        messages = []
        for event in r.get("transcript", {}).get("events", []):
            if "target" not in event.get("view", []):
                continue
            msg = event.get("edit", {}).get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "reasoning":
                        parts.append(block["reasoning"])
                    elif block.get("type") == "text":
                        parts.append(block.get("text", ""))
                content = "\n".join(parts)
            if role == "tool":
                role = "user"
            if role and content:
                messages.append({"role": role, "content": content})
        return messages
    return []


def has_bloom_contamination(messages: list[dict]) -> bool:
    """Check if messages contain BloomEvaluator or evaluator artifacts."""
    if not messages:
        return False
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            if any(marker in content for marker in [
                "BloomEvaluator",
                "<target_thinking>",
                "The behavior we are evaluating",
                "NO USER MODE",
                "The target model will execute",
            ]):
                return True
    return False


def fix_spans_with_contexts(dry_run: bool) -> int:
    path = DATA_DIR / "spans_with_contexts.json"
    with open(path) as f:
        spans = json.load(f)

    # Build span_text -> (behavior, variation) lookup from keyword_overlap source
    kw_path = PROJECT_ROOT / "probe_eval/results/v3_v2_5_combined_v1_span/keyword_overlap_raw.json"
    span_to_source = {}
    if kw_path.exists():
        with open(kw_path) as f:
            kw_raw = json.load(f)
        for item in kw_raw:
            text = item.get("text", "")[:500]
            span_to_source[text] = (item.get("behavior", ""), item.get("variation"))

    # Also load from context_dependency_raw for dev_fn spans
    ctx_dep_path = PROJECT_ROOT / "probe_eval/results/v3_v2_5_combined_v1_span/context_dependency_raw.json"
    if ctx_dep_path.exists():
        with open(ctx_dep_path) as f:
            ctx_raw = json.load(f)
        for item in ctx_raw:
            text = item.get("text", "")[:500]
            span_to_source[text] = (item.get("behavior", ""), item.get("variation"))

    fixed = 0
    for s in spans:
        msgs = s.get("messages", [])
        if not has_bloom_contamination(msgs):
            continue

        # Try direct fields first, then lookup
        behavior = s.get("behavior", "")
        variation = s.get("variation")
        if not behavior or variation is None:
            span_text = s.get("span_text", "")[:500]
            behavior, variation = span_to_source.get(span_text, ("", None))

        if not behavior or variation is None:
            continue

        target_msgs = load_target_messages(behavior, variation)
        if target_msgs:
            s["messages"] = target_msgs
            # Also save behavior/variation for future reference
            s["behavior"] = behavior
            s["variation"] = variation
            fixed += 1

    if not dry_run and fixed > 0:
        with open(path, "w") as f:
            json.dump(spans, f, indent=2)

    print(f"  spans_with_contexts.json: fixed {fixed}/{len(spans)}")
    return fixed


def fix_rollouts(dry_run: bool) -> int:
    """Rebuild positive_context rollout events from target-view messages."""
    path = DATA_DIR / "rollouts/rollout.json"
    with open(path) as f:
        data = json.load(f)

    # Also need spans_with_contexts to get behavior/variation for each rollout
    with open(DATA_DIR / "spans_with_contexts.json") as f:
        spans = json.load(f)
    valid_spans = [s for s in spans
                   if s.get("benign_context", {}).get("span_verbatim_check") == "PASSED"]

    fixed = 0
    for ri, r in enumerate(data["rollouts"]):
        cond = r["transcript"]["metadata"].get("condition", "")
        if cond != "positive_context":
            continue

        events = r["transcript"]["events"]
        # Check contamination
        has_contam = False
        for e in events:
            msg = e.get("edit", {}).get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                if "BloomEvaluator" in content or "<target_thinking>" in content:
                    has_contam = True
                    break
            elif isinstance(content, list):
                for block in content:
                    text = block.get("reasoning", "") + block.get("text", "")
                    if "BloomEvaluator" in text or "<target_thinking>" in text:
                        has_contam = True
                        break

        if not has_contam:
            continue

        # Find the corresponding span to get behavior/variation
        span_idx = ri // 3
        if span_idx >= len(valid_spans):
            continue
        span = valid_spans[span_idx]
        behavior = span.get("behavior", "")
        variation = span.get("variation")

        if not behavior or variation is None:
            continue

        target_msgs = load_target_messages(behavior, variation)
        if not target_msgs:
            continue

        # Rebuild events from target messages
        import uuid
        from datetime import datetime, timezone
        new_events = []
        span_text = r["transcript"]["metadata"].get("span_text", "")
        for msg in target_msgs:
            role = msg["role"]
            content = msg["content"]
            # For assistant turn containing span, split into reasoning + text
            if role == "assistant" and span_text in content:
                think_start = content.find("<think>")
                if think_start >= 0:
                    think_end = content.find("</think>")
                    if think_end >= 0:
                        reasoning = content[think_start + 7:think_end]
                        text = content[think_end + 8:].strip()
                        content = [
                            {"type": "reasoning", "reasoning": reasoning},
                            {"type": "text", "text": text},
                        ]

            new_events.append({
                "type": "transcript_event",
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "view": ["target", "combined"],
                "edit": {
                    "operation": "add",
                    "message": {
                        "role": role,
                        "id": str(uuid.uuid4()),
                        "content": content,
                    },
                },
            })

        r["transcript"]["events"] = new_events
        fixed += 1

    if not dry_run and fixed > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"  rollouts/rollout.json: fixed {fixed} rollouts")
    return fixed


def fix_training(dry_run: bool) -> int:
    """Fix training data by reloading messages from spans_with_contexts (which should be fixed first)."""
    # Load the already-fixed spans
    with open(DATA_DIR / "spans_with_contexts.json") as f:
        spans = json.load(f)

    # Build lookup: (indicator, span_text) -> fixed messages
    span_lookup = {}
    for s in spans:
        key = (s.get("indicator", ""), s.get("span_text", ""))
        span_lookup[key] = s.get("messages", [])

    total_fixed = 0
    for ds_type in ["context_only", "combined"]:
        ds_dir = DATA_DIR / "training" / ds_type
        if not ds_dir.exists():
            continue
        for fpath in sorted(ds_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)

            fixed = 0
            indicator = data.get("indicator_name", fpath.stem)
            for split in ["transcripts_train", "transcripts_val"]:
                for t in data.get(split, []):
                    msgs = t.get("messages", [])
                    if not has_bloom_contamination(msgs):
                        continue
                    # Look up fixed messages
                    span_text = t.get("span_text", "")
                    key = (indicator, span_text)
                    fixed_msgs = span_lookup.get(key)
                    if fixed_msgs:
                        t["messages"] = fixed_msgs
                        fixed += 1

            if not dry_run and fixed > 0:
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            if fixed > 0:
                print(f"  training/{ds_type}/{fpath.name}: fixed {fixed}")
            total_fixed += fixed

    return total_fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN\n")

    print("Fixing BloomEvaluator contamination (reloading from source rollouts):")

    # Fix in order: spans first (training references it)
    n1 = fix_spans_with_contexts(args.dry_run)
    n2 = fix_rollouts(args.dry_run)
    n3 = fix_training(args.dry_run)

    print(f"\nTotal: {n1 + n2 + n3} fixes")
    if args.dry_run:
        print("(rerun without --dry-run to apply)")


if __name__ == "__main__":
    main()
