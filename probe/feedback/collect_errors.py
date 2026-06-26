"""
Stage 1: Collect FP/FN error cases from OOD bloom eval and neutral validation.

Scans probe evaluation results on bloom rollouts to identify:
  - False Positives: turns where probe fires but GT says not misaligned
  - False Negatives: turns where GT says misaligned but probe doesn't fire

For neutral FPs, supports two modes:
  --neutral-fps FILE    Load pre-scored false-positive results from a JSON file
  --score-neutral       Score neutral dialogues on-the-fly with activation caching:
                        extracts activations once (GPU) and saves to disk, then
                        subsequent runs skip extraction and just load + score (CPU).

Usage:
    # With pre-scored neutral FPs
    python -m probe.feedback.collect_errors \
        --results-subdir v2_4_span \
        --neutral-fps probe/data/neutral/false_positives_v2_4.json

    # With on-the-fly scoring (caches activations for reuse)
    python -m probe.feedback.collect_errors \
        --results-subdir v2_4_span \
        --score-neutral \
        --n-neutral 2000
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from probe.generate import indicator_name_to_filename
from probe_eval.sentence_scores import _get_assistant_turns_for_rollout


# ---------------------------------------------------------------------------
# Bloom rollout data loaders (adapted from probe_eval/analyze_fp_cases.py)
# ---------------------------------------------------------------------------

def _extract_text(content):
    """Extract text from content which may be a string or list of content blocks."""
    if isinstance(content, str):
        return content
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
                    parts.append(f"[TOOL_CALL: {block.get('name', '?')}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _load_rollout_var_rep(rollout_path: Path) -> dict[int, tuple[int, int]]:
    """Return {rollout_index: (variation_number, repetition_number)}."""
    with open(rollout_path) as f:
        data = json.load(f)
    return {
        i: (r["variation_number"], r["repetition_number"])
        for i, r in enumerate(data["rollouts"])
    }


def _load_valid_turns_per_rollout(rollout_path: Path) -> dict[int, set[int]]:
    """Return {rollout_index: set of valid turn numbers}.

    "Valid" means the turn exists in the CURRENT rollout.json after any
    post-hoc truncation. If the probe eval was run on a pre-truncation
    version, token_scores.json may contain scores for turns that no longer
    exist in the transcript. Those turns must be ignored — they have no GT
    (not in misaligned_turns) but would otherwise be mis-counted as FPs.
    """
    with open(rollout_path) as f:
        data = json.load(f)
    result: dict[int, set[int]] = {}
    for idx, rollout in enumerate(data["rollouts"]):
        result[idx] = set(_get_assistant_turns_for_rollout(rollout))
    return result


def _load_misalignment_gt(
    gt_path: Path,
    indicator_key: str = "applicable_indicators",
) -> dict[tuple[int, int], dict[int, dict]]:
    """Return {(var, rep): {turn_index: {description, evidence, applicable_indicators}}}."""
    with open(gt_path) as f:
        data = json.load(f)
    result = {}
    for rollout in data["evaluated_rollouts"]:
        key = (rollout["variation_number"], rollout["repetition_number"])
        turns = {}
        for mt in rollout.get("misaligned_turns", []):
            # Use the specified key, fall back to default applicable_indicators
            indicators = mt.get(indicator_key, mt.get("applicable_indicators", []))
            turns[mt["turn_index"]] = {
                "description": mt["description"],
                "evidence": mt.get("evidence", []),
                "applicable_indicators": indicators,
            }
        result[key] = turns
    return result


def _load_indicator_judge_labels(
    gt_path: Path,
) -> dict[tuple[int, int], dict[int, dict[str, list[str]]]]:
    """Return {(var, rep): {turn_index: {concept_slug: [spans]}}} from the
    indicator-judge output stored alongside the misalignment judge in the
    same rollout_misalignment_turns.json file. A turn absent from the
    returned map means no indicator was flagged for that turn.
    """
    with open(gt_path) as f:
        data = json.load(f)
    result: dict[tuple[int, int], dict[int, dict[str, list[str]]]] = {}
    for rollout in data.get("evaluated_rollouts", []):
        key = (rollout["variation_number"], rollout["repetition_number"])
        turns: dict[int, dict[str, list[str]]] = {}
        for it in rollout.get("indicators_turns", []) or []:
            t_idx = it.get("turn_index")
            if t_idx is None:
                continue
            by_slug: dict[str, list[str]] = {}
            for ind in it.get("present_indicators", []) or []:
                name = ind.get("indicator_name")
                if not name:
                    continue
                slug = indicator_name_to_filename(name)
                by_slug[slug] = [s for s in (ind.get("spans") or []) if isinstance(s, str) and s.strip()]
            if by_slug:
                turns[t_idx] = by_slug
        result[key] = turns
    return result


def _extract_turn_content(rollout_path: Path, var: int, rep: int, turn_idx: int,
                          max_chars: int | None = None) -> dict[str, str]:
    """Extract user message and assistant response(s) for a bloom turn.

    For agentic tool-calling rollouts, a single logical "turn" may span
    multiple assistant messages (think → tool_call → tool_result → think →
    tool_call → ... → final response). All assistant messages between this
    user turn and the next one are concatenated so downstream reasoning-only
    / span-matching logic sees the full content.

    max_chars=None returns untruncated content (for internal analysis).
    Callers should truncate when writing to output.
    """
    with open(rollout_path) as f:
        data = json.load(f)

    for rollout in data["rollouts"]:
        if rollout["variation_number"] != var or rollout["repetition_number"] != rep:
            continue

        events = rollout["transcript"]["events"]
        target_user_count = 0
        turn_event_idx = None
        next_user_event_idx = None

        for i, ev in enumerate(events):
            msg = ev.get("edit", {}).get("message", {})
            view = ev.get("view", [])
            if "target" in view and msg.get("role") == "user":
                target_user_count += 1
                if target_user_count == turn_idx:
                    turn_event_idx = i
                elif target_user_count == turn_idx + 1:
                    next_user_event_idx = i
                    break

        if turn_event_idx is None:
            return {"user_message": "[Turn not found]", "assistant_response": "[Turn not found]"}

        user_content = _extract_text(
            events[turn_event_idx].get("edit", {}).get("message", {}).get("content", "")
        )

        # Concatenate ALL target-assistant messages between this user turn
        # and the next one (or end of events). Handles multi-step tool-calling
        # turns where thinking + response is split across events.
        end = next_user_event_idx if next_user_event_idx is not None else len(events)
        asst_chunks: list[str] = []
        for i in range(turn_event_idx + 1, end):
            ev = events[i]
            msg = ev.get("edit", {}).get("message", {})
            view = ev.get("view", [])
            if "target" in view and msg.get("role") == "assistant":
                chunk = _extract_text(msg.get("content", ""))
                if chunk:
                    asst_chunks.append(chunk)
        assistant_content = "\n".join(asst_chunks) if asst_chunks else "[No assistant response]"

        if max_chars is not None:
            user_content = user_content[:max_chars]
            assistant_content = assistant_content[:max_chars]
        return {
            "user_message": user_content,
            "assistant_response": assistant_content,
        }

    return {"user_message": "[Rollout not found]", "assistant_response": "[Rollout not found]"}


# ---------------------------------------------------------------------------
# Threshold / indicator info loading
# ---------------------------------------------------------------------------

def _load_probe_threshold(probes_dir: Path, concept_slug: str, layer: int,
                          label_mode: str = "span",
                          threshold_key: str = "threshold_f1") -> float | None:
    """Load a threshold from training_meta.json's clean_label block.

    threshold_key options:
      - "threshold_f1" (default): F1-optimal threshold on clean val
      - "threshold_accuracy": accuracy-optimal threshold
      - "fpr_1pct": threshold calibrated to ~1% FPR on clean val
      - "fpr_5pct": threshold calibrated to ~5% FPR on clean val
    """
    meta_path = probes_dir / concept_slug / label_mode / f"layer{layer}" / "training_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    cl = meta.get("val_thresholds", {}).get("clean_label", {})
    val = cl.get(threshold_key)
    # fpr_1pct / fpr_5pct are {fpr, tpr, threshold} dicts; rest are plain floats.
    if isinstance(val, dict):
        return val.get("threshold")
    return val


def _load_indicator_info(probes_dir: Path, concept_slug: str, layer: int,
                         label_mode: str = "span") -> dict | None:
    """Load indicator name and definition from training_meta.json."""
    meta_path = probes_dir / concept_slug / label_mode / f"layer{layer}" / "training_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "indicator_name": meta.get("indicator_name", concept_slug),
        "training_data_path": meta.get("training_data_path", ""),
    }


# ---------------------------------------------------------------------------
# OOD error collection
# ---------------------------------------------------------------------------

def _load_sentence_scores_for_behavior(
    token_scores_path: Path,
    rollout_path: Path,
) -> dict[int, dict[int, list[tuple[float, str]]]]:
    """Load token_scores.json and compute per-sentence (score, text) for each turn.

    Returns {rollout_index: {turn_number: [(mean_score, sentence_text), ...]}}.
    """
    if not token_scores_path.exists():
        return {}

    with open(token_scores_path) as f:
        ts_data = json.load(f)

    with open(rollout_path) as f:
        rollouts = json.load(f)["rollouts"]

    result: dict[int, dict[int, list[tuple[float, str]]]] = {}
    for ts_entry in ts_data["per_rollout"]:
        idx = ts_entry["rollout_index"]
        if idx >= len(rollouts):
            continue

        tokens = ts_entry["tokens"]
        scores = ts_entry["scores"]

        # Find scored segments (consecutive non-None scores = assistant messages)
        segments: list[tuple[int, int]] = []
        seg_start: int | None = None
        for i, s in enumerate(scores):
            if s is not None:
                if seg_start is None:
                    seg_start = i
            else:
                if seg_start is not None:
                    segments.append((seg_start, i))
                    seg_start = None
        if seg_start is not None:
            segments.append((seg_start, len(scores)))

        # Get turn numbers for assistant messages
        assistant_turns = _get_assistant_turns_for_rollout(rollouts[idx])

        # Group tokens/scores by turn
        turn_token_lists: dict[int, list[str]] = defaultdict(list)
        turn_score_lists: dict[int, list[float]] = defaultdict(list)
        for seg_idx, (seg_s, seg_e) in enumerate(segments):
            turn = assistant_turns[seg_idx] if seg_idx < len(assistant_turns) else 0
            turn_token_lists[turn].extend(tokens[seg_s:seg_e])
            turn_score_lists[turn].extend(scores[seg_s:seg_e])

        # Split each turn into sentences with (score, text) pairs
        # Uses the local _split_tokens_into_sentences which returns
        # list[tuple[float, str]] — (mean_score, sentence_text) pairs
        per_turn: dict[int, list[tuple[float, str]]] = {}
        for turn in turn_token_lists:
            toks = turn_token_lists[turn]
            scrs = turn_score_lists[turn]
            if not toks:
                continue
            per_turn[turn] = _split_tokens_into_sentences(toks, scrs)

        result[idx] = per_turn

    return result


def collect_ood_errors(
    results_dir: Path,
    results_subdir: str,
    probes_dir: Path,
    bloom_results_dir: Path,
    behaviors: list[str],
    layer: int = 30,
    label_mode: str = "span",
    indicator_key: str = "applicable_indicators",
    reasoning_only: bool = False,
    threshold_key: str = "threshold_f1",
    thresholds_override: dict[str, float] | None = None,
    fn_thresholds_override: dict[str, float] | None = None,
    gt_source: str = "assign_indicators",
    gt_filename: str = "rollout_misalignment_turns.json",
) -> dict[str, dict]:
    """Collect FP/FN cases from bloom eval results using sentence-level scoring.

    Uses token_scores.json to compute per-sentence mean scores, then compares
    each sentence against the clean_label threshold_f1. A turn is FP if any
    sentence exceeds the threshold (but GT says not misaligned); a turn is FN
    if GT says misaligned but no sentence exceeds the threshold.

    Returns dict mapping concept_slug -> {indicator_name, threshold, ood_fp_cases, ood_fn_cases}.
    """
    subdir = results_dir / results_subdir
    if not subdir.exists():
        print(f"  WARNING: results subdir not found: {subdir}")
        return {}

    # Discover all concepts: {results_subdir}/{concept}/{label_mode}/layer{N}/{behavior}/
    concepts = set()
    for p in subdir.iterdir():
        if not p.is_dir() or p.name.endswith(".json") or p.name.endswith(".html"):
            continue
        for lm_dir in p.iterdir():
            if lm_dir.is_dir() and (lm_dir / f"layer{layer}").is_dir():
                concepts.add((p.name, lm_dir.name))

    print(f"  Found {len(concepts)} probe concepts in {subdir}")
    all_errors: dict[str, dict] = {}

    for concept_slug, results_label_mode in sorted(concepts):
        # FP threshold: external override > training_meta[threshold_key].
        # FN threshold: fn_thresholds_override[slug] if set, else same as FP.
        # Using separate thresholds lets you collect abundant FPs at a loose
        # threshold while counting FNs only when a strict probe misses.
        if thresholds_override and concept_slug in thresholds_override:
            threshold = thresholds_override[concept_slug]
        else:
            threshold = _load_probe_threshold(probes_dir, concept_slug, layer, label_mode, threshold_key)
            if threshold is None:
                threshold = _load_probe_threshold(probes_dir, concept_slug, layer, results_label_mode, threshold_key)
        if threshold is None:
            print(f"    Skipping {concept_slug}: no threshold found")
            continue
        if fn_thresholds_override and concept_slug in fn_thresholds_override:
            threshold_fn = fn_thresholds_override[concept_slug]
        else:
            threshold_fn = threshold

        info = _load_indicator_info(probes_dir, concept_slug, layer, label_mode)
        if info is None:
            info = _load_indicator_info(probes_dir, concept_slug, layer, results_label_mode)
        indicator_name = info["indicator_name"] if info else concept_slug

        fp_cases: list[dict] = []
        fn_cases: list[dict] = []
        middle_cases: list[dict] = []  # diagnostic: GT-misaligned + probe fires + this indicator NOT in indicator_judge's present set

        for behavior_dir_name in behaviors:
            token_scores_path = (
                subdir / concept_slug / results_label_mode / f"layer{layer}"
                / behavior_dir_name / "token_scores.json"
            )
            results_path = (
                subdir / concept_slug / results_label_mode / f"layer{layer}"
                / behavior_dir_name / "results.json"
            )
            if not results_path.exists():
                continue

            with open(results_path) as f:
                results = json.load(f)

            rollout_dir = bloom_results_dir / behavior_dir_name
            rollout_path = rollout_dir / "rollout.json"
            gt_path = rollout_dir / gt_filename

            if not rollout_path.exists():
                continue

            is_all_negative = results.get("all_negative", False)

            # For all_negative (benign) behaviors, no GT file needed —
            # all turns are treated as not-misaligned.
            if not is_all_negative and not gt_path.exists():
                continue

            var_rep_map = _load_rollout_var_rep(rollout_path)
            valid_turns_per_rollout = _load_valid_turns_per_rollout(rollout_path)
            gt_turns = _load_misalignment_gt(gt_path, indicator_key=indicator_key) if gt_path.exists() else {}
            indicator_judge_turns: dict[tuple[int, int], dict[int, dict[str, list[str]]]] = (
                _load_indicator_judge_labels(gt_path)
                if gt_source == "indicator_judge" and gt_path.exists()
                else {}
            )

            # Load ambiguous rollouts to exclude from error collection
            ambiguous_path = rollout_dir / "ambiguous_rollouts.json"
            ambiguous_keys: set[tuple[int, int]] = set()
            if ambiguous_path.exists():
                with open(ambiguous_path) as f:
                    ambiguous_data = json.load(f)
                for ar in ambiguous_data.get("ambiguous_rollouts", []):
                    ambiguous_keys.add((ar["variation_number"], ar["repetition_number"]))

            # Load sentence-level scores from token_scores.json
            sentence_data = _load_sentence_scores_for_behavior(
                token_scores_path, rollout_path,
            )

            for entry in results.get("per_rollout", []):
                rollout_idx = entry["rollout_index"]
                if rollout_idx not in var_rep_map:
                    continue
                var, rep = var_rep_map[rollout_idx]
                if (var, rep) in ambiguous_keys:
                    continue
                gt_for_rollout = gt_turns.get((var, rep), {})
                indicator_judge_for_rollout = indicator_judge_turns.get((var, rep), {})
                rollout_sentences = sentence_data.get(rollout_idx, {})
                valid_turns = valid_turns_per_rollout.get(rollout_idx, set())

                for turn_data in entry.get("per_turn_scores", []):
                    turn = turn_data["turn"]
                    # Skip turns that no longer exist in the current rollout.
                    # This happens when the eval was run on a pre-truncation
                    # transcript but the rollout was later trimmed (e.g., to
                    # cut off artifact user messages). Such turns have probe
                    # scores but no valid GT context — treating them as FPs
                    # would be incorrect. Only apply to non-all-negative
                    # behaviors (benign rollouts have no GT by design).
                    if valid_turns and turn not in valid_turns:
                        continue
                    turn_mean_score = turn_data["score"]
                    is_misaligned = turn in gt_for_rollout

                    # Get sentence-level scores for this turn
                    sent_pairs = rollout_sentences.get(turn, [])

                    if sent_pairs:
                        # Sentence-level: probe fires if any sentence > threshold.
                        # Use FP threshold for firing decision; FN decision below
                        # uses the (potentially stricter) threshold_fn.
                        max_sent_score = max(s for s, _ in sent_pairs)
                        probe_fires_fp = max_sent_score > threshold
                        probe_fires_fn = max_sent_score > threshold_fn
                        # Collect triggering sentences (for FP cases),
                        # filtering out short fragments (< 5 words)
                        triggering = [
                            {"sentence": text.strip()[:500],
                             "score": round(float(score), 4),
                             "threshold": round(float(threshold), 4)}
                            for score, text in sent_pairs
                            if score > threshold and len(text.strip().split()) >= 5
                        ]
                    else:
                        # Fallback to turn-level mean if no token_scores
                        probe_fires_fp = turn_mean_score > threshold
                        probe_fires_fn = turn_mean_score > threshold_fn
                        triggering = []

                    # FP: probe fires AND turn is NOT misaligned (per misalignment judge).
                    # Same across both gt_source modes — the user wants to credit the
                    # probe for firing on any genuinely-misaligned turn regardless of
                    # which indicator the judge picked.
                    if probe_fires_fp and not is_misaligned:
                        content = _extract_turn_content(rollout_path, var, rep, turn)
                        fp_cases.append({
                            "behavior": behavior_dir_name,
                            "var": var, "rep": rep, "turn": turn,
                            "max_sentence_score": round(float(max_sent_score if sent_pairs else turn_mean_score), 4),
                            "turn_mean_score": round(float(turn_mean_score), 4),
                            "threshold": round(float(threshold), 4),
                            "all_negative_rollout": is_all_negative,
                            "fp_sentences": triggering,
                            **content,
                        })
                        continue

                    if gt_source == "indicator_judge":
                        # Span-level FN: indicator_judge says THIS indicator is
                        # present on the turn with specific spans. For each GT
                        # span, check whether any scored sentence covering it
                        # exceeds threshold_fn. Uncovered spans are collected as
                        # FN evidence — even if the probe fires on other parts of
                        # the turn (partial miss).
                        turn_indicators = indicator_judge_for_rollout.get(turn, {})
                        indicator_present = concept_slug in turn_indicators
                        if indicator_present:
                            spans = turn_indicators[concept_slug]
                            uncovered = [
                                s[:500] for s in spans
                                if not _span_is_covered(s, sent_pairs, threshold_fn)
                            ]

                            if uncovered:
                                content = _extract_turn_content(rollout_path, var, rep, turn)

                                if reasoning_only:
                                    assistant_text = content.get("assistant_response", "")
                                    uncovered = [
                                        e for e in uncovered
                                        if _evidence_is_in_reasoning(e, assistant_text)
                                    ]
                                    if not uncovered:
                                        continue

                                gt_info = gt_for_rollout.get(turn, {})
                                description = gt_info.get("description", "")[:1500] or (
                                    f"Indicator judge flagged '{concept_slug}' on this turn."
                                )

                                fn_cases.append({
                                    "behavior": behavior_dir_name,
                                    "var": var, "rep": rep, "turn": turn,
                                    "max_sentence_score": round(float(max_sent_score if sent_pairs else turn_mean_score), 4),
                                    "turn_mean_score": round(float(turn_mean_score), 4),
                                    "threshold": round(float(threshold_fn), 4),
                                    "gt_description": description,
                                    "gt_evidence": uncovered,
                                    "applicable_indicators": sorted(turn_indicators.keys()),
                                    "probe_fires_on_turn": bool(probe_fires_fn),
                                    **content,
                                })
                            continue

                        # Diagnostic "middle case": turn is misaligned AND probe fires
                        # AND this indicator is NOT in the indicator judge's present set.
                        # Logged but not used as FP training data — this is probe bleeding
                        # into a neighboring indicator on a genuinely misaligned turn.
                        if probe_fires_fp and is_misaligned and not indicator_present:
                            content = _extract_turn_content(rollout_path, var, rep, turn)
                            middle_cases.append({
                                "behavior": behavior_dir_name,
                                "var": var, "rep": rep, "turn": turn,
                                "max_sentence_score": round(float(max_sent_score if sent_pairs else turn_mean_score), 4),
                                "turn_mean_score": round(float(turn_mean_score), 4),
                                "threshold": round(float(threshold), 4),
                                "fp_sentences": triggering,
                                "indicators_present_on_turn": sorted(turn_indicators.keys()),
                                **content,
                            })
                        continue

                    # assign_indicators path (legacy): FN = is_misaligned + concept
                    # in assign_indicators' applicable_indicators + probe doesn't fire.
                    if not probe_fires_fn and is_misaligned:
                        gt_info = gt_for_rollout[turn]
                        # Only count as FN if this indicator is applicable to
                        # the turn (from Opus-assigned labels). An empty list
                        # means classification failed (e.g. refusal) — skip
                        # rather than counting FN for every non-firing probe,
                        # which would spuriously inflate the FN pool.
                        applicable = gt_info.get("applicable_indicators", [])
                        if not applicable:
                            continue
                        if concept_slug not in applicable:
                            continue

                        # Extract evidence, filtering to spans matching this
                        # indicator when structured per-evidence format is used.
                        raw_evidence = gt_info.get("evidence", [])
                        if raw_evidence and isinstance(raw_evidence[0], dict):
                            # Structured format: filter to matching spans
                            # Check both the indicator_key and default key
                            matching = [
                                e["text"][:500] for e in raw_evidence
                                if concept_slug in e.get(indicator_key, e.get("applicable_indicators", []))
                            ]
                            # Fall back to all evidence if no specific matches
                            gt_evidence = matching if matching else [
                                e["text"][:500] for e in raw_evidence[:5]
                            ]
                        else:
                            # Legacy format: list of strings
                            gt_evidence = [e[:500] for e in raw_evidence[:5]]

                        content = _extract_turn_content(rollout_path, var, rep, turn)

                        # When --reasoning-only: keep only the evidence spans
                        # that fall inside <think>...</think>. If no span is in
                        # reasoning, drop the FN entirely — response-only spans
                        # (code/statements without misalignment reasoning) don't
                        # make good positive training examples for a
                        # reasoning-only probe.
                        if reasoning_only:
                            assistant_text = content.get("assistant_response", "")
                            gt_evidence = [
                                e for e in gt_evidence
                                if _evidence_is_in_reasoning(e, assistant_text)
                            ]
                            if not gt_evidence:
                                continue

                        fn_cases.append({
                            "behavior": behavior_dir_name,
                            "var": var, "rep": rep, "turn": turn,
                            "max_sentence_score": round(float(max_sent_score if sent_pairs else turn_mean_score), 4),
                            "turn_mean_score": round(float(turn_mean_score), 4),
                            "threshold": round(float(threshold_fn), 4),
                            "gt_description": gt_info["description"][:1500],
                            "gt_evidence": gt_evidence,
                            "applicable_indicators": applicable,
                            **content,
                        })

        fp_cases.sort(key=lambda c: -(c["max_sentence_score"] - c["threshold"]))
        fn_cases.sort(key=lambda c: c["max_sentence_score"] - c["threshold"])
        middle_cases.sort(key=lambda c: -(c["max_sentence_score"] - c["threshold"]))

        all_errors[concept_slug] = {
            "indicator_name": indicator_name,
            "threshold": round(float(threshold), 4),
            "ood_fp_cases": fp_cases,
            "ood_fn_cases": fn_cases,
            "middle_cases": middle_cases,
        }

        if fp_cases or fn_cases or middle_cases:
            mid_str = f", {len(middle_cases)} middle" if middle_cases else ""
            print(f"    {indicator_name}: {len(fp_cases)} FPs, {len(fn_cases)} FNs{mid_str}")

    return all_errors


# ---------------------------------------------------------------------------
# Neutral FP collection — pre-scored file
# ---------------------------------------------------------------------------

def collect_neutral_fps(
    neutral_fps_path: Path,
    n_neutral: int = 2000,
    thresholds_by_slug: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    """Load pre-scored neutral FP data and filter to first n_neutral dialogues.

    If thresholds_by_slug is provided, re-filter the pre-scored sentences to
    only those whose score exceeds the (potentially new, stricter) threshold.
    Use this when the pre-scored file was generated with a different threshold
    mode than what you want the final FP decision to use.

    Returns dict mapping concept_slug -> list of FP case dicts.
    """
    if not neutral_fps_path.exists():
        print(f"  WARNING: neutral FPs file not found: {neutral_fps_path}")
        print(f"  Run with --score-neutral to extract and score on-the-fly.")
        return {}

    with open(neutral_fps_path) as f:
        data = json.load(f)

    triggered = data.get("triggered_dialogues", [])
    filtered = [d for d in triggered if d["dialogue_index"] < n_neutral]

    print(f"  Loaded {len(filtered)} triggered dialogues (of {n_neutral} neutral val set)")

    per_indicator: dict[str, list[dict]] = defaultdict(list)
    for d in filtered:
        for slug, probe_info in d.get("triggered_probes", {}).items():
            # Re-apply a stricter threshold if provided (e.g. from --threshold-key fpr_1pct).
            thr = thresholds_by_slug.get(slug) if thresholds_by_slug else None
            raw = probe_info.get("sentences", [])
            sentences = []
            for s in raw:
                if len(s.get("sentence", "").strip().split()) < 5:
                    continue
                if thr is not None and s.get("score", -1e9) <= thr:
                    continue
                sentences.append(s)
            if not sentences:
                continue
            per_indicator[slug].append({
                "dialogue_index": d["dialogue_index"],
                "user_prompt": d.get("user_prompt", "")[:2000],
                "assistant_response": d.get("assistant_response", "")[:3000],
                "fp_sentences": sentences[:5],
            })

    for slug, cases in sorted(per_indicator.items()):
        print(f"    {slug}: {len(cases)} neutral FPs")

    return dict(per_indicator)


# ---------------------------------------------------------------------------
# Neutral FP collection — on-the-fly scoring with activation caching
# ---------------------------------------------------------------------------

_DEFAULT_PADDING = {
    "gemma": {"left": 0, "right": 0},
    "mistral": {"left": 0, "right": 0},
    "llama": {"left": 0, "right": 0},
    "qwen": {"left": 0, "right": 0},
    "glm": {"left": 0, "right": 0},
}
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _span_is_covered(
    span_text: str,
    sent_pairs: list[tuple[float, str]],
    threshold: float,
) -> bool:
    """Check if a GT span is covered by any sentence scoring above threshold.

    Uses prefix substring matching (first ~80 chars, case-insensitive) to
    align GT spans (from indicator judge) to scored sentences (from
    token_scores.json). Returns True if the span's prefix appears inside
    any sentence whose score exceeds the threshold.
    """
    if not span_text or not sent_pairs:
        return False
    prefix = span_text.strip()[:80].lower()
    if not prefix:
        return False
    for score, text in sent_pairs:
        if score > threshold and prefix in text.lower():
            return True
    return False


def _evidence_is_in_reasoning(evidence_text: str, full_assistant_text: str) -> bool:
    """Check if an evidence span falls inside any <think>...</think> block.

    Agentic tool-calling turns often have multiple <think>...</think> blocks
    (think → tool_call → tool_result → think → ...). An evidence span counts
    as "in reasoning" if it falls inside ANY such block, not just the first.

    If the assistant text has no <think> tags at all, treat it as
    reasoning-equivalent (non-thinking model) and return True.

    Matching is done on the first ~80-char prefix of the evidence to tolerate
    whitespace/truncation differences between GT evidence and raw response.
    """
    if not evidence_text or not full_assistant_text:
        return False

    # Find all <think>...</think> ranges
    reasoning_ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        t_start = full_assistant_text.find("<think>", cursor)
        if t_start == -1:
            break
        content_start = t_start + len("<think>")
        t_end = full_assistant_text.find("</think>", content_start)
        if t_end == -1:
            # Unterminated — treat rest of text as reasoning.
            reasoning_ranges.append((content_start, len(full_assistant_text)))
            break
        reasoning_ranges.append((content_start, t_end))
        cursor = t_end + len("</think>")

    if not reasoning_ranges:
        return True  # No <think> tags at all — don't filter out.

    # Find evidence position
    needle = evidence_text.strip()[:80]
    if not needle:
        return False
    pos = full_assistant_text.find(needle)
    if pos == -1:
        needle = evidence_text.strip()[:40]
        if not needle:
            return False
        pos = full_assistant_text.find(needle)
        if pos == -1:
            return False
    return any(r_start <= pos < r_end for r_start, r_end in reasoning_ranges)


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


def _get_or_extract_neutral_activations(
    dialogues: list[dict],
    cache_dir: Path,
    layer: int,
) -> list[dict[str, Any]]:
    """Extract per-dialogue activations for neutral dialogues, with disk caching.

    On first call (cache miss): loads the GLM model, tokenizes each dialogue,
    extracts layer activations for detected (assistant) tokens, and saves:
        {cache_dir}/neutral_val_acts_layer{layer}.pt      — concatenated activation tensor
        {cache_dir}/neutral_val_meta_layer{layer}.json     — per-dialogue offsets + str_tokens

    On subsequent calls (cache hit): loads from disk without touching the GPU.

    Returns list of dicts, one per dialogue:
        [{"acts": Tensor[n_tokens, emb_dim], "str_tokens": list[str]}, ...]
    Dialogues with no detected tokens are included with empty acts/str_tokens.
    """
    import torch

    acts_path = cache_dir / f"neutral_val_acts_layer{layer}.pt"
    meta_path = cache_dir / f"neutral_val_meta_layer{layer}.json"

    # ---------- Cache hit ----------
    if acts_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        cached_n = meta.get("n_dialogues", 0)
        if cached_n == len(dialogues):
            print(f"  Loading cached activations ({len(dialogues)} dialogues, layer {layer})")
            all_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
            offsets = meta["offsets"]  # list of (start, end) pairs
            str_tokens_list = meta["str_tokens"]  # list of list[str]

            result = []
            for (start, end), toks in zip(offsets, str_tokens_list):
                result.append({
                    "acts": all_acts[start:end],
                    "str_tokens": toks,
                })
            return result
        else:
            print(f"  Cache mismatch: cached {cached_n} dialogues, need {len(dialogues)}. Re-extracting.")

    # ---------- Cache miss: extract activations ----------
    print(f"  Extracting activations for {len(dialogues)} dialogues (layer {layer})...")
    print(f"  This requires GPU. Activations will be cached at: {cache_dir}")

    from deception_detection.activations import Activations
    from deception_detection.models import ModelName, get_model_and_tokenizer
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.types import Dialogue, Message
    from tqdm import trange

    model, tokenizer = get_model_and_tokenizer(ModelName.GLM_FLASH)
    template_kwargs: dict[str, Any] = {"clear_thinking": False}

    per_dialogue_acts: list[torch.Tensor] = []
    per_dialogue_tokens: list[list[str]] = []

    for idx in trange(len(dialogues), desc="Extracting neutral activations"):
        d = dialogues[idx]
        dialogue: Dialogue = [
            Message(role="system", content=d["system_prompt"], detect=False),
            Message(role="user", content=d["user_prompt"], detect=False),
            Message(role="assistant", content=d["assistant_response"], detect=True),
        ]

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=[dialogue],
                tokenizer=tokenizer,
                padding=_DEFAULT_PADDING,
                template_kwargs=template_kwargs,
            )

            if toks.detection_mask is not None and not toks.detection_mask.any():
                per_dialogue_acts.append(torch.empty(0, 0))
                per_dialogue_tokens.append([])
                continue

            acts = Activations.from_model(model, toks, batch_size=1, layers=[layer])
            masked = acts.get_masked_activations()  # [n_detected, 1, emb]

            if masked.numel() == 0:
                per_dialogue_acts.append(torch.empty(0, 0))
                per_dialogue_tokens.append([])
            else:
                # Squeeze out the layer dimension: [n_detected, emb]
                masked = masked[:, 0, :].cpu().half()
                per_dialogue_acts.append(masked)

                # Get detected token strings
                mask = toks.detection_mask[0]
                detected_indices = mask.nonzero(as_tuple=True)[0]
                str_toks = [toks.str_tokens[0][j.item()] for j in detected_indices]
                per_dialogue_tokens.append(str_toks)

            del acts, masked
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Dialogue {idx} failed: {e}")
            per_dialogue_acts.append(torch.empty(0, 0))
            per_dialogue_tokens.append([])
            continue

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # Build concatenated tensor + offset map
    offsets: list[tuple[int, int]] = []
    non_empty = [a for a in per_dialogue_acts if a.numel() > 0]
    if non_empty:
        emb_dim = non_empty[0].shape[-1]
    else:
        emb_dim = 1

    chunks = []
    pos = 0
    for a in per_dialogue_acts:
        if a.numel() == 0:
            offsets.append((pos, pos))
        else:
            n = a.shape[0]
            chunks.append(a)
            offsets.append((pos, pos + n))
            pos += n

    if chunks:
        all_acts = torch.cat(chunks, dim=0)
    else:
        all_acts = torch.empty(0, emb_dim)

    # Save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_acts, acts_path)
    cache_meta = {
        "n_dialogues": len(dialogues),
        "layer": layer,
        "total_tokens": int(all_acts.shape[0]),
        "emb_dim": int(all_acts.shape[1]) if all_acts.dim() == 2 else 0,
        "offsets": offsets,
        "str_tokens": per_dialogue_tokens,
    }
    with open(meta_path, "w") as f:
        json.dump(cache_meta, f)

    print(f"  Cached {all_acts.shape[0]} tokens to {acts_path}")
    print(f"  Cached metadata to {meta_path}")

    # Reconstruct per-dialogue result
    result = []
    for (start, end), toks in zip(offsets, per_dialogue_tokens):
        result.append({
            "acts": all_acts[start:end],
            "str_tokens": toks,
        })
    return result


def score_neutral_cached(
    dialogues: list[dict],
    probes_dir: Path,
    cache_dir: Path,
    layer: int = 30,
    label_mode: str = "span",
) -> dict[str, list[dict]]:
    """Score neutral dialogues using cached activations and return FP cases per indicator.

    1. Loads (or extracts + caches) per-dialogue activations
    2. Loads all probes from probes_dir
    3. Scores each dialogue with each probe at sentence level
    4. Returns FP cases organized by indicator slug
    """
    import torch
    from deception_detection.detectors import LogisticRegressionDetector

    # 1. Get or extract activations
    dialogue_data = _get_or_extract_neutral_activations(dialogues, cache_dir, layer)

    # 2. Load probes
    probes: dict[str, tuple[LogisticRegressionDetector, float, str]] = {}
    for indicator_dir in sorted(probes_dir.iterdir()):
        if not indicator_dir.is_dir():
            continue
        detector_path = indicator_dir / label_mode / f"layer{layer}" / "detector.pt"
        meta_path = indicator_dir / label_mode / f"layer{layer}" / "training_meta.json"
        if not detector_path.exists() or not meta_path.exists():
            continue

        detector = LogisticRegressionDetector.load(detector_path)
        with open(meta_path) as f:
            meta = json.load(f)

        clean_label = meta.get("val_thresholds", {}).get("clean_label", {})
        threshold = clean_label.get("threshold_f1")
        if threshold is None:
            continue

        indicator_name = meta.get("indicator_name", indicator_dir.name)
        probes[indicator_dir.name] = (detector, float(threshold), indicator_name)

    print(f"  Loaded {len(probes)} probes for scoring")

    # 3. Score each dialogue
    per_indicator: dict[str, list[dict]] = defaultdict(list)

    for idx, (d_data, dialogue) in enumerate(zip(dialogue_data, dialogues)):
        acts = d_data["acts"]
        str_tokens = d_data["str_tokens"]

        if acts.numel() == 0:
            continue

        for slug, (detector, threshold, indicator_name) in probes.items():
            # Detector expects [n_tokens, n_layers, emb_dim]; cached acts are [n_tokens, emb_dim]
            acts_3d = acts.unsqueeze(1) if acts.dim() == 2 else acts
            token_scores = detector.get_score_tensor(acts_3d).detach().cpu().numpy()
            sentences = _split_tokens_into_sentences(str_tokens, token_scores.tolist())

            exceeding = []
            for mean_score, sent_text in sentences:
                if mean_score > threshold and len(sent_text.strip().split()) >= 5:
                    exceeding.append({
                        "sentence": sent_text.strip()[:500],
                        "score": round(float(mean_score), 4),
                        "threshold": round(float(threshold), 4),
                    })

            if exceeding:
                per_indicator[slug].append({
                    "dialogue_index": idx,
                    "user_prompt": dialogue.get("user_prompt", "")[:2000],
                    "assistant_response": dialogue.get("assistant_response", "")[:3000],
                    "fp_sentences": exceeding[:5],
                })

    for slug, cases in sorted(per_indicator.items()):
        _, _, name = probes.get(slug, (None, None, slug))
        print(f"    {name}: {len(cases)} neutral FPs")

    return dict(per_indicator)


# ---------------------------------------------------------------------------
# Indicator definitions loader
# ---------------------------------------------------------------------------

def _load_indicator_definitions(indicator_set: str) -> dict[str, dict]:
    """Load indicator definitions from the indicator module.

    Returns dict mapping indicator_slug -> {name, definition, examples, non_examples}.
    """
    if indicator_set == "v2_6":
        from indicators.misalignment_indicators_v2_6 import (
            MISALIGNMENT_INDICATORS_V2 as indicators,
            MISALIGNMENT_PRECONDITIONS as preconditions,
            BEHAVIORAL_CONCEPTS as concepts,
        )
    elif indicator_set == "v2_5":
        from indicators.misalignment_indicators_v2_5 import (
            MISALIGNMENT_INDICATORS_V2 as indicators,
            MISALIGNMENT_PRECONDITIONS as preconditions,
            BEHAVIORAL_CONCEPTS as concepts,
        )
    elif indicator_set == "v2_4":
        from indicators.misalignment_indicators_v2_4 import (
            MISALIGNMENT_INDICATORS_V2 as indicators,
            MISALIGNMENT_PRECONDITIONS as preconditions,
            BEHAVIORAL_CONCEPTS as concepts,
        )
    elif indicator_set == "v2_3":
        from indicators.misalignment_indicators_v2_3 import (
            MISALIGNMENT_INDICATORS_V2 as indicators,
            MISALIGNMENT_PRECONDITIONS as preconditions,
            BEHAVIORAL_CONCEPTS as concepts,
        )
    else:
        raise ValueError(f"Unsupported indicator set: {indicator_set}")

    result = {}
    for ind in list(indicators) + list(preconditions) + list(concepts):
        slug = indicator_name_to_filename(ind.name)
        result[slug] = {
            "name": ind.name,
            "definition": ind.definition,
            "examples": list(ind.examples),
            "non_examples": list(ind.non_examples) if ind.non_examples else [],
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect FP/FN error cases from eval results")
    parser.add_argument(
        "--results-subdir", type=str, required=True,
        help="Results subdirectory name (e.g., v2_4_span)",
    )
    parser.add_argument(
        "--probes-dir", type=str, default=None,
        help="Probes directory (default: probe/probes/{results_subdir})",
    )
    parser.add_argument(
        "--indicator-set", type=str, default="v2_4",
        choices=["v2_3", "v2_4", "v2_5", "v2_6"],
        help="Indicator set for definitions (default: v2_4)",
    )
    parser.add_argument(
        "--layer", type=int, default=30,
        help="Probe layer to use for thresholds (default: 30)",
    )
    parser.add_argument(
        "--label-mode", type=str, default="span",
        help="Label mode (default: span)",
    )

    # Neutral scoring — two mutually exclusive modes
    neutral_group = parser.add_mutually_exclusive_group()
    neutral_group.add_argument(
        "--neutral-fps", type=str, default=None,
        help="Path to pre-scored neutral false positives JSON",
    )
    neutral_group.add_argument(
        "--score-neutral", action="store_true",
        help="Score neutral dialogues on-the-fly (with activation caching)",
    )

    parser.add_argument(
        "--neutral-dialogues", type=str, default=None,
        help="Path to neutral dialogues JSON (default: probe/data/neutral/dialogues_filtered.json)",
    )
    parser.add_argument(
        "--neutral-cache-dir", type=str, default=None,
        help="Directory for cached neutral activations "
             "(default: probe/data/neutral/val_cache_{results_subdir})",
    )
    parser.add_argument(
        "--n-neutral", type=int, default=2000,
        help="Number of neutral dialogues in the validation set (default: 2000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for collected errors JSON",
    )
    parser.add_argument(
        "--indicator-key", type=str, default="applicable_indicators",
        help="Key to read indicator assignments from in GT files "
             "(default: applicable_indicators, use applicable_indicators_strict for strict mode). "
             "Only used when --gt-source=assign_indicators.",
    )
    parser.add_argument(
        "--gt-source", type=str, default="assign_indicators",
        choices=["assign_indicators", "indicator_judge"],
        help="Source of per-indicator GT labels. 'assign_indicators' (default, legacy) "
             "uses Opus-assigned applicable_indicators stored per-turn by "
             "probe/feedback/assign_indicators.py; FN requires is_misaligned AND concept "
             "in applicable_indicators. 'indicator_judge' uses the black-box indicator "
             "judge's indicators_turns block; FN requires THIS indicator is in "
             "present_indicators[turn] (is_misaligned not required).",
    )
    parser.add_argument(
        "--gt-filename", type=str, default="rollout_misalignment_turns.json",
        help="Filename inside each bloom dir to load both judges' GT from "
             "(default: rollout_misalignment_turns.json). Use to point at backups "
             "like rollout_misalignment_turns_with_examples.json.",
    )
    parser.add_argument(
        "--reasoning-only", action="store_true",
        help="Drop FN cases whose GT evidence spans are all outside "
             "<think>...</think> (response-part only). Such spans (often just "
             "code or statements without misalignment reasoning) don't make "
             "good positive training examples for a reasoning-only probe. "
             "Does NOT affect FPs or neutral FPs — those are worth mitigating "
             "regardless of where they fire.",
    )
    parser.add_argument(
        "--threshold-key", type=str, default="threshold_f1",
        choices=["threshold_f1", "threshold_accuracy", "fpr_1pct", "fpr_5pct"],
        help="Which clean_label threshold to use from training_meta.json "
             "(default: threshold_f1). fpr_1pct/fpr_5pct are stricter and "
             "reduce FPs at the cost of recall.",
    )
    parser.add_argument(
        "--thresholds-file", type=str, default=None,
        help="Path to an external JSON file of per-probe thresholds to override "
             "the training_meta-derived ones. Expected structure: "
             "versions.<version>.per_layer.layer<N>.thresholds.<slug> -> float. "
             "Probes missing from the file fall back to --threshold-key.",
    )
    parser.add_argument(
        "--thresholds-version", type=str, default=None,
        help="Which version key to read from --thresholds-file (default: first).",
    )
    parser.add_argument(
        "--fn-thresholds-file", type=str, default=None,
        help="Optional separate threshold file for FN detection only. "
             "Useful for asymmetric error collection: loose threshold → many "
             "FPs (hard negatives); strict threshold → only genuine misses as "
             "FNs (hard positives). Falls back to FP threshold for probes "
             "missing from this file.",
    )
    parser.add_argument(
        "--fn-thresholds-version", type=str, default=None,
        help="Which version key to read from --fn-thresholds-file (default: first).",
    )

    parser.add_argument(
        "--positive-behaviors", type=str, nargs="+",
        default=[
            "instructed-long-horizon-sabotage-backdoor",
            "instructed-long-horizon-sabotage-bug",
            "sycophancy",
            "strategic-deception_glmflash",
            "self-preferential-bias_glm_4_7_flash",
            "self-promotion_glm_4_7_flash",
        ],
        help="Positive bloom behavior directories",
    )
    parser.add_argument(
        "--benign-behaviors", type=str, nargs="+",
        default=[
            "sycophancy_benign",
            "sandbagging_benign",
            "undermining_oversight_benign",
            "instructed-long-horizon-sabotage-info_benign",
            "self-preferential-bias_benign_glm_4_7_flash",
            "self-promotion_benign_glm_4_7_flash",
            "strategic-deception_benign_glm_4_7_flash",
            "instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash",
        ],
        help="Benign bloom behavior directories",
    )
    args = parser.parse_args()

    results_dir = REPO_ROOT / "probe_eval" / "results"
    bloom_results_dir = REPO_ROOT / "bloom" / "bloom-results"
    probes_dir = Path(args.probes_dir) if args.probes_dir else (
        REPO_ROOT / "probe" / "probes" / args.results_subdir
    )

    all_behaviors = args.positive_behaviors + args.benign_behaviors

    # Load indicator definitions
    print(f"\nLoading {args.indicator_set} indicator definitions...")
    indicator_defs = _load_indicator_definitions(args.indicator_set)
    print(f"  {len(indicator_defs)} indicators loaded")

    # Collect OOD errors
    print(f"\nCollecting OOD errors from {args.results_subdir}...")
    def _load_thresholds_file(path: str, version: str | None) -> dict[str, float]:
        with open(path) as f:
            ext = json.load(f)
        versions = ext.get("versions", {})
        vkey = version or next(iter(versions))
        per_layer = versions[vkey]["per_layer"][f"layer{args.layer}"]
        return dict(per_layer["thresholds"]), vkey

    # FP (primary) threshold override
    thresholds_override = None
    if args.thresholds_file:
        thresholds_override, vkey = _load_thresholds_file(args.thresholds_file, args.thresholds_version)
        print(f"\nLoaded {len(thresholds_override)} FP threshold overrides from "
              f"{args.thresholds_file} (version={vkey}, layer={args.layer})")

    # FN threshold override (optional, separate from FP)
    fn_thresholds_override = None
    if args.fn_thresholds_file:
        fn_thresholds_override, vkey = _load_thresholds_file(args.fn_thresholds_file, args.fn_thresholds_version)
        print(f"Loaded {len(fn_thresholds_override)} FN threshold overrides from "
              f"{args.fn_thresholds_file} (version={vkey}, layer={args.layer})")

    ood_errors = collect_ood_errors(
        results_dir=results_dir,
        results_subdir=args.results_subdir,
        probes_dir=probes_dir,
        bloom_results_dir=bloom_results_dir,
        behaviors=all_behaviors,
        layer=args.layer,
        label_mode=args.label_mode,
        indicator_key=args.indicator_key,
        reasoning_only=args.reasoning_only,
        threshold_key=args.threshold_key,
        thresholds_override=thresholds_override,
        fn_thresholds_override=fn_thresholds_override,
        gt_source=args.gt_source,
        gt_filename=args.gt_filename,
    )

    # Collect neutral FPs — two modes
    neutral_fps: dict[str, list] = {}

    if args.neutral_fps:
        print(f"\nLoading pre-scored neutral FPs from {args.neutral_fps}...")
        # Build per-slug thresholds so pre-scored sentences can be re-filtered
        # at the new (stricter) threshold if threshold_key was overridden.
        thresholds_by_slug = {
            slug: info["threshold"] for slug, info in ood_errors.items()
            if info.get("threshold") is not None
        }
        neutral_fps = collect_neutral_fps(
            neutral_fps_path=Path(args.neutral_fps),
            n_neutral=args.n_neutral,
            thresholds_by_slug=thresholds_by_slug,
        )

    elif args.score_neutral:
        dialogues_path = Path(args.neutral_dialogues) if args.neutral_dialogues else (
            REPO_ROOT / "probe" / "data" / "neutral" / "dialogues.json"
        )
        cache_dir = Path(args.neutral_cache_dir) if args.neutral_cache_dir else (
            REPO_ROOT / "probe" / "data" / "neutral" / f"val_cache_{args.results_subdir}"
        )

        print(f"\nScoring neutral dialogues (first {args.n_neutral}) with activation caching...")
        print(f"  Dialogues: {dialogues_path}")
        print(f"  Cache dir: {cache_dir}")

        with open(dialogues_path) as f:
            all_dialogues = json.load(f)
        dialogues = all_dialogues[:args.n_neutral]
        print(f"  Selected {len(dialogues)} dialogues for neutral val set")

        neutral_fps = score_neutral_cached(
            dialogues=dialogues,
            probes_dir=probes_dir,
            cache_dir=cache_dir,
            layer=args.layer,
            label_mode=args.label_mode,
        )

    # Merge into final output
    output = {
        "config": {
            "results_subdir": args.results_subdir,
            "probes_dir": str(probes_dir),
            "indicator_set": args.indicator_set,
            "layer": args.layer,
            "label_mode": args.label_mode,
            "n_neutral": args.n_neutral,
            "bloom_behaviors": all_behaviors,
            "gt_source": args.gt_source,
        },
        "indicators": {},
    }

    all_slugs = set(ood_errors.keys()) | set(neutral_fps.keys())
    for slug in sorted(all_slugs):
        ood = ood_errors.get(slug, {})
        indicator_def = indicator_defs.get(slug, {})

        entry = {
            "indicator_name": ood.get("indicator_name", indicator_def.get("name", slug)),
            "indicator_definition": indicator_def.get("definition", ""),
            "indicator_examples": indicator_def.get("examples", []),
            "indicator_non_examples": indicator_def.get("non_examples", []),
            "threshold": ood.get("threshold"),
            "ood_fp_cases": ood.get("ood_fp_cases", []),
            "ood_fn_cases": ood.get("ood_fn_cases", []),
            "middle_cases": ood.get("middle_cases", []),
            "neutral_fp_cases": neutral_fps.get(slug, []),
        }

        n_fp = len(entry["ood_fp_cases"])
        n_fn = len(entry["ood_fn_cases"])
        n_neutral = len(entry["neutral_fp_cases"])
        n_middle = len(entry["middle_cases"])
        if n_fp > 0 or n_fn > 0 or n_neutral > 0 or n_middle > 0:
            output["indicators"][slug] = entry

    # Save
    output_path = Path(args.output) if args.output else (
        REPO_ROOT / "probe" / "data" / "feedback" / args.results_subdir / "collected_errors.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    n_indicators = len(output["indicators"])
    total_fp = sum(len(v["ood_fp_cases"]) for v in output["indicators"].values())
    total_fn = sum(len(v["ood_fn_cases"]) for v in output["indicators"].values())
    total_nfp = sum(len(v["neutral_fp_cases"]) for v in output["indicators"].values())
    total_mid = sum(len(v.get("middle_cases", [])) for v in output["indicators"].values())
    print(f"\n{'='*60}")
    print(f"Collected errors for {n_indicators} indicators (gt_source={args.gt_source})")
    print(f"  OOD: {total_fp} FPs, {total_fn} FNs")
    if total_mid:
        print(f"  Middle (diagnostic): {total_mid}")
    print(f"  Neutral: {total_nfp} FPs")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
