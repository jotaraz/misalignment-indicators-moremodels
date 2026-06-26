"""Collect FP/FN transcripts directly from bloom eval & neutral FP scoring
as hard-negative/hard-positive training examples, then merge with original data.

This is a simpler alternative to the full feedback loop (Opus analysis + Sonnet
generation).  Instead of generating *new* synthetic data, we reuse the actual
mis-scored transcripts:

  - Bloom FP  → negative training example  (probe fired, but GT says benign)
  - Bloom FN  → positive training example  (probe missed real misalignment)
  - Neutral FP → negative training example  (probe fired on benign conversation)

For FN (positive) examples, span labels come from the Opus misalignment judge's
ground-truth evidence (``rollout_misalignment_turns.json`` and the ``gt_evidence``
field in ``collected_errors.json``).  Evidence strings are fuzzy-matched against
the full conversation text to find exact substrings for span-mode training.

Usage:
    python -m probe.feedback.collect_direct_examples \
        --collected-errors probe/data/feedback/v2_4_span_v3/collected_errors.json \
        --neutral-fps probe/data/neutral/false_positives_v2_4_combined_v3_span_dev.json \
        --original-data-dir probe/data/v2_4 \
        --output-dir probe/data/v2_4_direct_v1
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers for extracting full conversations from bloom rollouts
# ---------------------------------------------------------------------------

def _strip_tool_call_tags(text: str) -> str:
    """Remove tool-call XML blocks from text.

    The GLM chat template treats tags like <tool_call>, <function_call>,
    <function_calls>, <invoke> etc. as special tokens, causing assertion
    errors during tokenization. We strip them since probe training only needs
    the reasoning/text content.
    """
    # Remove matched tool-related XML blocks (may span multiple lines)
    for tag in ("tool_call", "function_call", "function_calls", "invoke",
                "tool", "arguments"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
    # Remove unclosed tool_call/function_call tags (truncated at end)
    text = re.sub(r"<(?:tool_call|function_call|function_calls|invoke)>.*$",
                  "", text, flags=re.DOTALL)
    return text


def _extract_text(content) -> str:
    """Convert content (string or list of content blocks) to plain text.

    Includes tool_use input content (code written by the model) so that
    evidence strings referencing code output can be matched as spans.
    Strips <tool_call> XML tags that break the chat template tokenizer.
    """
    if isinstance(content, str):
        return _strip_tool_call_tags(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "reasoning":
                    r = block.get("reasoning", "")
                    if r:
                        parts.append(f"<think>{r}</think>")
                elif block.get("type") == "tool_use":
                    # Include tool_use input so code evidence can be matched
                    inp = block.get("input", {})
                    tool_parts = [f"[TOOL_CALL: {block.get('name', '?')}]"]
                    if isinstance(inp, dict):
                        for v in inp.values():
                            if isinstance(v, str):
                                tool_parts.append(v)
                    elif isinstance(inp, str):
                        tool_parts.append(inp)
                    parts.append("\n".join(tool_parts))
            elif isinstance(block, str):
                parts.append(block)
        return _strip_tool_call_tags("\n".join(parts))
    return _strip_tool_call_tags(str(content))


def _extract_full_conversation(
    rollout_path: Path,
    var: int,
    rep: int,
    up_to_turn: int,
) -> list[dict[str, str]] | None:
    """Extract the full conversation from a bloom rollout up to *up_to_turn*.

    Returns a list of ``{role, content}`` dicts (system + user/assistant pairs),
    or ``None`` if the rollout is not found.
    """
    with open(rollout_path) as f:
        data = json.load(f)

    for rollout in data["rollouts"]:
        if rollout["variation_number"] != var or rollout["repetition_number"] != rep:
            continue

        events = rollout["transcript"]["events"]
        system_prompt = rollout["transcript"].get("metadata", {}).get(
            "target_system_prompt", "You are a helpful assistant."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        user_turn_count = 0

        for ev in events:
            msg = ev.get("edit", {}).get("message", {})
            view = ev.get("view", [])
            if "target" not in view:
                continue
            role = msg.get("role")
            if role == "user":
                user_turn_count += 1
                if user_turn_count > up_to_turn:
                    break
                messages.append({"role": "user", "content": _extract_text(msg.get("content", ""))})
            elif role == "assistant":
                if user_turn_count >= 1 and user_turn_count <= up_to_turn:
                    text = _extract_text(msg.get("content", ""))
                    # Merge consecutive assistant messages (tool-use splits)
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["content"] += "\n" + text
                    else:
                        messages.append({"role": "assistant", "content": text})

        return messages if len(messages) > 1 else None

    return None


def _extract_think_block(text: str) -> str | None:
    """Return the content of the first <think>...</think> block, or None."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Loading misalignment judge ground-truth
# ---------------------------------------------------------------------------

def _load_misalignment_turns(behavior_dir: Path) -> dict | None:
    """Load rollout_misalignment_turns.json for a behavior directory."""
    mt_path = behavior_dir / "rollout_misalignment_turns.json"
    if not mt_path.exists():
        return None
    with open(mt_path) as f:
        return json.load(f)


def _get_judge_evidence(
    misalignment_turns_data: dict | None,
    var: int,
    rep: int,
    turn: int,
    indicator_slug: str,
) -> list[str]:
    """Get evidence strings from the Opus judge for a specific (var, rep, turn, indicator).

    Returns evidence strings from the ``evidence`` list of matching misaligned
    turns, filtered to turns that list the indicator in ``applicable_indicators``.

    Supports both structured evidence (list of dicts with per-evidence indicators)
    and legacy format (list of strings).
    """
    if not misalignment_turns_data:
        return []

    evidence: list[str] = []
    for rollout in misalignment_turns_data.get("evaluated_rollouts", []):
        if rollout.get("variation_number") != var or rollout.get("repetition_number") != rep:
            continue
        for mt in rollout.get("misaligned_turns", []):
            if mt.get("turn_index") != turn:
                continue
            if indicator_slug not in mt.get("applicable_indicators", []):
                continue
            raw_evidence = mt.get("evidence", [])
            if raw_evidence and isinstance(raw_evidence[0], dict):
                # Structured format: only include spans matching this indicator
                matching = [
                    e["text"] for e in raw_evidence
                    if indicator_slug in e.get("applicable_indicators", [])
                ]
                # Fall back to all evidence if no specific matches
                evidence.extend(matching if matching else [
                    e["text"] for e in raw_evidence
                ])
            else:
                # Legacy format: list of strings
                evidence.extend(raw_evidence)
    return evidence


# ---------------------------------------------------------------------------
# Fuzzy span matching
# ---------------------------------------------------------------------------

def _fuzzy_find_span(
    needle: str,
    haystack: str,
    threshold: float = 0.55,
) -> str | None:
    """Find the best approximate substring match of *needle* in *haystack*.

    Uses ``difflib.SequenceMatcher`` with a sliding window.  Returns the
    matched substring from *haystack*, or ``None`` if no match exceeds
    *threshold*.
    """
    if not needle or not haystack:
        return None

    # Normalize escaped newlines
    needle_norm = needle.replace("\\n", "\n").strip()

    # Exact substring check first (fast path)
    if needle_norm in haystack:
        return needle_norm

    n_len = len(needle_norm)
    if n_len < 5:
        return None

    best_ratio = 0.0
    best_start = 0
    best_end = 0

    # Try windows of varying sizes around the needle length
    for win_mult in [1.0, 0.8, 1.2, 0.7, 1.5]:
        win_len = max(10, int(n_len * win_mult))
        step = max(1, win_len // 4)
        for i in range(0, len(haystack) - win_len + 1, step):
            candidate = haystack[i : i + win_len]
            ratio = SequenceMatcher(None, needle_norm, candidate, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + win_len

    if best_ratio >= threshold:
        return haystack[best_start:best_end]
    return None


def _find_gt_spans(
    messages: list[dict[str, str]],
    indicator_turn: int,
    gt_evidence: list[str],
    judge_evidence: list[str],
    threshold: float = 0.55,
) -> list[str]:
    """Find ground-truth spans in the assistant message at *indicator_turn*.

    Tries to fuzzy-match each evidence string (from ``collected_errors.json``
    and ``rollout_misalignment_turns.json``) against the target assistant
    message.  Falls back to the ``<think>`` block if no evidence matches.

    Returns a list of matched span strings (may be empty).
    """
    # Find the assistant message at indicator_turn (1-based)
    assistant_count = 0
    target_content = ""
    for msg in messages:
        if msg["role"] == "assistant":
            assistant_count += 1
            if assistant_count == indicator_turn:
                target_content = msg["content"]
                break

    if not target_content:
        return []

    # Collect all candidate evidence strings, deduplicated
    all_evidence: list[str] = []
    seen: set[str] = set()
    for ev in gt_evidence + judge_evidence:
        ev_key = ev.replace("\\n", "\n").strip()
        if ev_key and ev_key not in seen:
            seen.add(ev_key)
            all_evidence.append(ev)

    # Try to match each evidence string against the message content
    matched_spans: list[str] = []
    matched_set: set[str] = set()
    for ev in all_evidence:
        span = _fuzzy_find_span(ev, target_content, threshold=threshold)
        if span and span not in matched_set:
            matched_spans.append(span)
            matched_set.add(span)

    # Fall back to <think> block if no evidence matched
    if not matched_spans:
        think = _extract_think_block(target_content)
        if think:
            matched_spans = [think]

    return matched_spans


# ---------------------------------------------------------------------------
# Building training transcripts from error cases
# ---------------------------------------------------------------------------

def _build_transcript_from_messages(
    messages: list[dict[str, str]],
    indicator_turn: int | None,
    transcript_id: str,
    label: str,
    scenario: str = "",
    spans: list[str] | None = None,
) -> dict[str, Any]:
    """Build a training transcript dict from a list of messages.

    Args:
        messages: [{role, content}, ...] — system + user/assistant pairs.
        indicator_turn: 1-based turn number where the indicator is present,
            or None for all-negative transcripts.
        transcript_id: Unique ID for this transcript.
        label: "deceptive" or "honest".
        scenario: Brief description.
        spans: Pre-computed span strings for the indicator turn.
    """
    per_turn_labels = []
    assistant_count = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            assistant_count += 1
            entry: dict[str, Any] = {"turn_index": i, "has_indicator": False}
            if indicator_turn is not None and assistant_count == indicator_turn:
                entry["has_indicator"] = True
                if spans:
                    entry["spans"] = spans
            per_turn_labels.append(entry)

    return {
        "id": transcript_id,
        "label": label,
        "scenario_description": scenario,
        "messages": messages,
        "per_turn_labels": per_turn_labels,
    }


def _build_neutral_transcript(
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
    transcript_id: str,
) -> dict[str, Any]:
    """Build a negative training transcript from a neutral dialogue."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
    return {
        "id": transcript_id,
        "label": "honest",
        "scenario_description": "neutral conversation (hard negative from FP scoring)",
        "messages": messages,
        "per_turn_labels": [{"turn_index": 2, "has_indicator": False}],
    }


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect_direct_examples(
    collected_errors_path: Path,
    neutral_fps_path: Path | None,
    neutral_dialogues_path: Path | None,
    original_data_dir: Path,
    output_dir: Path,
    max_bloom_fp: int = 100,
    max_bloom_fn: int = 100,
    max_neutral_fp: int = 100,
) -> dict[str, dict[str, int]]:
    """Collect direct FP/FN examples and merge with original training data.

    Returns a dict of {indicator_slug: {bloom_fp, bloom_fn, neutral_fp, original, total}}.
    """
    with open(collected_errors_path) as f:
        errors = json.load(f)

    # Load neutral FP data
    neutral_fp_data: dict | None = None
    if neutral_fps_path and neutral_fps_path.exists():
        with open(neutral_fps_path) as f:
            neutral_fp_data = json.load(f)

    # Load neutral dialogues (for full text — FP JSON may truncate)
    neutral_dialogues: list[dict] | None = None
    if neutral_dialogues_path and neutral_dialogues_path.exists():
        with open(neutral_dialogues_path) as f:
            neutral_dialogues = json.load(f)

    # Build bloom behavior → rollout dir mapping
    # Use cwd as base (scripts are run from the project root)
    base_dir = Path.cwd()
    behavior_rollout_dirs: dict[str, Path] = {}
    for slug, ind_data in errors.get("indicators", {}).items():
        for case in ind_data.get("ood_fp_cases", []) + ind_data.get("ood_fn_cases", []):
            behavior = case["behavior"]
            if behavior not in behavior_rollout_dirs:
                p = base_dir / f"bloom/bloom-results/{behavior}" / "rollout.json"
                if p.exists():
                    behavior_rollout_dirs[behavior] = p

    # Scan bloom-results for any we missed
    bloom_results = base_dir / "bloom" / "bloom-results"
    if bloom_results.exists():
        for d in bloom_results.iterdir():
            if d.is_dir() and (d / "rollout.json").exists():
                behavior_rollout_dirs.setdefault(d.name, d / "rollout.json")

    # Load misalignment turns data per behavior (for GT evidence)
    misalignment_turns_cache: dict[str, dict | None] = {}
    for behavior, rollout_path in behavior_rollout_dirs.items():
        behavior_dir = rollout_path.parent
        if behavior not in misalignment_turns_cache:
            misalignment_turns_cache[behavior] = _load_misalignment_turns(behavior_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    span_stats = {"total_fn": 0, "with_gt_spans": 0, "with_think_fallback": 0, "no_spans": 0}

    for slug, ind_data in sorted(errors.get("indicators", {}).items()):
        indicator_name = ind_data["indicator_name"]
        print(f"\n{'='*60}")
        print(f"Indicator: {indicator_name} ({slug})")

        # Load original training data
        original_path = original_data_dir / f"{slug}.json"
        if not original_path.exists():
            print(f"  SKIP: no original data at {original_path}")
            continue
        with open(original_path) as f:
            original = json.load(f)

        new_transcripts: list[dict] = []

        # ---- Bloom FP cases → negative examples ----
        fp_cases = ind_data.get("ood_fp_cases", [])[:max_bloom_fp]
        n_bloom_fp = 0
        for case in fp_cases:
            behavior = case["behavior"]
            rollout_path = behavior_rollout_dirs.get(behavior)
            if not rollout_path:
                continue
            messages = _extract_full_conversation(
                rollout_path, case["var"], case["rep"], case["turn"],
            )
            if not messages:
                continue
            tid = f"{slug}_bloom_fp_{behavior}_v{case['var']}_r{case['rep']}_t{case['turn']}"
            transcript = _build_transcript_from_messages(
                messages, indicator_turn=None, transcript_id=tid,
                label="honest",
                scenario=f"bloom FP from {behavior} (score={case.get('max_sentence_score', 0):.2f})",
            )
            new_transcripts.append(transcript)
            n_bloom_fp += 1

        # ---- Bloom FN cases → positive examples (with GT spans) ----
        fn_cases = ind_data.get("ood_fn_cases", [])[:max_bloom_fn]
        n_bloom_fn = 0
        for case in fn_cases:
            behavior = case["behavior"]
            rollout_path = behavior_rollout_dirs.get(behavior)
            if not rollout_path:
                continue
            messages = _extract_full_conversation(
                rollout_path, case["var"], case["rep"], case["turn"],
            )
            if not messages:
                continue

            # Collect GT evidence from collected_errors + misalignment judge
            gt_evidence = case.get("gt_evidence", [])
            judge_evidence = _get_judge_evidence(
                misalignment_turns_cache.get(behavior),
                case["var"], case["rep"], case["turn"], slug,
            )

            # Fuzzy-match evidence against the conversation
            spans = _find_gt_spans(
                messages, case["turn"], gt_evidence, judge_evidence,
            )

            # Track span stats
            span_stats["total_fn"] += 1
            if spans and not (len(spans) == 1 and _extract_think_block(spans[0])):
                span_stats["with_gt_spans"] += 1
            elif spans:
                span_stats["with_think_fallback"] += 1
            else:
                span_stats["no_spans"] += 1

            tid = f"{slug}_bloom_fn_{behavior}_v{case['var']}_r{case['rep']}_t{case['turn']}"
            transcript = _build_transcript_from_messages(
                messages, indicator_turn=case["turn"], transcript_id=tid,
                label="deceptive",
                scenario=f"bloom FN from {behavior} (score={case.get('max_sentence_score', 0):.2f})",
                spans=spans if spans else None,
            )
            new_transcripts.append(transcript)
            n_bloom_fn += 1

        # ---- Neutral FP cases → negative examples ----
        n_neutral_fp = 0
        if neutral_fp_data and neutral_dialogues:
            for triggered in neutral_fp_data.get("triggered_dialogues", []):
                if n_neutral_fp >= max_neutral_fp:
                    break
                if slug not in triggered.get("triggered_probes", {}):
                    continue

                didx = triggered["dialogue_index"]
                if didx < len(neutral_dialogues):
                    dlg = neutral_dialogues[didx]
                    tid = f"{slug}_neutral_fp_d{didx}"
                    transcript = _build_neutral_transcript(
                        system_prompt=dlg.get("system_prompt", "You are a helpful assistant."),
                        user_prompt=dlg["user_prompt"],
                        assistant_response=dlg["assistant_response"],
                        transcript_id=tid,
                    )
                    new_transcripts.append(transcript)
                    n_neutral_fp += 1

        n_original = len(original.get("transcripts", []))
        print(f"  bloom FP: {n_bloom_fp}, bloom FN: {n_bloom_fn}, neutral FP: {n_neutral_fp}")
        print(f"  original: {n_original}, new: {len(new_transcripts)}, total: {n_original + len(new_transcripts)}")

        # Merge
        combined = copy.deepcopy(original)
        combined["transcripts"].extend(new_transcripts)

        # Save
        out_path = output_dir / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  Saved: {out_path}")

        stats[slug] = {
            "bloom_fp": n_bloom_fp,
            "bloom_fn": n_bloom_fn,
            "neutral_fp": n_neutral_fp,
            "original": n_original,
            "total": n_original + len(new_transcripts),
        }

    # Save stats summary
    stats["_span_matching"] = span_stats
    stats_path = output_dir / "_direct_feedback_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSpan matching stats: {span_stats}")
    print(f"Stats saved to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect FP/FN transcripts as direct training examples"
    )
    parser.add_argument(
        "--collected-errors", type=str, required=True,
        help="Path to collected_errors.json from feedback collect step",
    )
    parser.add_argument(
        "--neutral-fps", type=str, default=None,
        help="Path to neutral false_positives JSON",
    )
    parser.add_argument(
        "--neutral-dialogues", type=str, default=None,
        help="Path to neutral dialogues.json (for full text)",
    )
    parser.add_argument(
        "--original-data-dir", type=str, required=True,
        help="Path to original training data dir (e.g., probe/data/v2_4)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to write combined training data",
    )
    parser.add_argument("--max-bloom-fp", type=int, default=100)
    parser.add_argument("--max-bloom-fn", type=int, default=100)
    parser.add_argument("--max-neutral-fp", type=int, default=100)

    args = parser.parse_args()

    collect_direct_examples(
        collected_errors_path=Path(args.collected_errors),
        neutral_fps_path=Path(args.neutral_fps) if args.neutral_fps else None,
        neutral_dialogues_path=Path(args.neutral_dialogues) if args.neutral_dialogues else None,
        original_data_dir=Path(args.original_data_dir),
        output_dir=Path(args.output_dir),
        max_bloom_fp=args.max_bloom_fp,
        max_bloom_fn=args.max_bloom_fn,
        max_neutral_fp=args.max_neutral_fp,
    )


if __name__ == "__main__":
    main()
