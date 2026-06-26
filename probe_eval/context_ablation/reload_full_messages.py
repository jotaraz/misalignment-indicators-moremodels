"""Temporary script: reload full (non-truncated) messages into spans_with_contexts.json.

The original select_and_generate.py truncated message content to 500 chars,
causing 65% of positive_context rollouts to lose their span text. This script
reloads full messages from the training data and original rollouts, then
rebuilds the rollout files.

Usage:
    python -m probe_eval.context_ablation.reload_full_messages
"""

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
TRAINING_DATA_DIR = PROJECT_ROOT / "probe/data/v3_v2_5_combined_v1"
BLOOM_RESULTS_DIR = PROJECT_ROOT / "bloom/bloom-results"


def load_training_transcripts() -> dict[str, list[dict]]:
    """Load all training transcripts indexed by transcript_id -> messages."""
    tid_to_messages = {}
    for path in TRAINING_DATA_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        for t in data["transcripts"]:
            tid = t.get("id", "")
            if tid and t.get("label") == "deceptive":
                tid_to_messages[tid] = t.get("messages", [])
    return tid_to_messages


def load_rollout_messages(behavior: str, variation: int, turn: int | None = None) -> list[dict]:
    """Load full messages from a bloom rollout file."""
    rollout_path = BLOOM_RESULTS_DIR / behavior / "rollout.json"
    if not rollout_path.exists():
        return []
    with open(rollout_path) as f:
        data = json.load(f)
    for r in data.get("rollouts", []):
        if r.get("variation_number") == variation:
            events = r.get("transcript", {}).get("events", [])
            messages = []
            for e in events:
                # Only include target-view messages (skip evaluator harness)
                if "target" not in e.get("view", []):
                    continue
                msg = e.get("edit", {}).get("message", {})
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get("type") == "reasoning":
                            parts.append(block["reasoning"])
                        elif block.get("type") == "text":
                            parts.append(block["text"])
                    content = "\n".join(parts)
                if role == "tool":
                    role = "user"
                messages.append({"role": role, "content": content})
            return messages
    return []


def main():
    spans_path = DATA_DIR / "spans_with_contexts.json"
    with open(spans_path) as f:
        spans = json.load(f)
    print(f"Loaded {len(spans)} spans")

    # Count initial state
    n_found_before = sum(
        1 for s in spans
        if any(s["span_text"].lower() in m.get("content", "").lower()
               for m in s.get("messages", []))
    )
    print(f"Span in messages before: {n_found_before}/{len(spans)}")

    # Load training transcript lookup
    tid_to_messages = load_training_transcripts()
    print(f"Loaded {len(tid_to_messages)} training transcripts")

    # Reload messages
    n_reloaded = 0
    n_failed = 0
    by_source = defaultdict(lambda: [0, 0])  # [reloaded, failed]

    for span in spans:
        source = span.get("source", "")
        tid = span.get("transcript_id", "")
        span_text = span.get("span_text", "")

        # Check if already has full messages (span text found)
        msgs = span.get("messages", [])
        if any(span_text.lower() in m.get("content", "").lower() for m in msgs):
            continue  # Already good

        # Source 1: Training spans — match by transcript_id
        if source == "training" and tid in tid_to_messages:
            full_msgs = tid_to_messages[tid]
            # Verify span text is in the full messages
            if any(span_text.lower() in m["content"].lower() for m in full_msgs):
                span["messages"] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in full_msgs
                ]
                n_reloaded += 1
                by_source[source][0] += 1
                continue

        # Source 2: Dev spans — reload from rollout files
        if source in ("dev_fired_tp", "dev_fired_fp", "dev_fn_context_dependent"):
            behavior = span.get("behavior", "")
            variation = span.get("variation")
            turn = span.get("turn")
            if behavior and variation is not None:
                full_msgs = load_rollout_messages(behavior, variation, turn)
                if full_msgs and any(
                    span_text.lower() in m["content"].lower() for m in full_msgs
                ):
                    span["messages"] = full_msgs
                    n_reloaded += 1
                    by_source[source][0] += 1
                    continue

        n_failed += 1
        by_source[source][1] += 1

    # Also remove turn_text truncation
    for span in spans:
        # Reload turn_text from the message at turn_index if available
        turn_idx = span.get("turn_index")
        msgs = span.get("messages", [])
        if turn_idx is not None and turn_idx < len(msgs):
            span["turn_text"] = msgs[turn_idx]["content"]

    n_found_after = sum(
        1 for s in spans
        if any(s["span_text"].lower() in m.get("content", "").lower()
               for m in s.get("messages", []))
    )

    print(f"\nReloaded: {n_reloaded}, Failed: {n_failed}")
    for src, (ok, fail) in sorted(by_source.items()):
        print(f"  {src}: {ok} reloaded, {fail} failed")
    print(f"Span in messages after: {n_found_after}/{len(spans)} (was {n_found_before})")

    # Save updated spans
    backup_path = DATA_DIR / "spans_with_contexts_truncated_backup.json"
    import shutil
    shutil.copy2(spans_path, backup_path)
    print(f"\nBackup -> {backup_path}")

    with open(spans_path, "w") as f:
        json.dump(spans, f, indent=2)
    print(f"Updated -> {spans_path}")

    # Rebuild rollouts
    print("\n=== Rebuilding rollouts ===")
    from probe_eval.context_ablation.build_rollouts import main as build_main
    build_main()


if __name__ == "__main__":
    main()
