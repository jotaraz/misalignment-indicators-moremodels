"""Per-sentence scoring helpers for probe evaluation.

Computes sentence-level average scores from token-level scores, used by
both indicator and misalignment ground-truth evaluation.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _split_tokens_into_sentences(
    tokens: list[str], scores: list[float],
    short_sentence_mode: str = "merge",
    min_words: int = 5,
) -> list[float]:
    """Split a scored token sequence into sentences and return per-sentence mean scores.

    Sentence boundaries: sentence-ending punctuation followed by whitespace,
    or double newlines.  Falls back to a single sentence if no boundary found.

    Args:
        short_sentence_mode: How to handle sentences with fewer than *min_words* words.
            "merge" (default): merge into adjacent sentence.
            "discard": drop them entirely (they don't contribute to the max).
    """
    if not tokens:
        return []

    text = "".join(tokens)

    # Character positions where new sentences begin
    split_positions = sorted(set(m.end() for m in _SENTENCE_SPLIT_RE.finditer(text)))

    # Walk through tokens, advancing sentence index when a token's midpoint
    # crosses a split position.
    sentence_lists: list[list[float]] = [[]]
    sentence_token_lists: list[list[str]] = [[]]
    char_pos = 0
    split_idx = 0

    for i, tok in enumerate(tokens):
        tok_mid = char_pos + len(tok) / 2
        while split_idx < len(split_positions) and tok_mid >= split_positions[split_idx]:
            split_idx += 1
            sentence_lists.append([])
            sentence_token_lists.append([])
        sentence_lists[-1].append(scores[i])
        sentence_token_lists[-1].append(tok)
        char_pos += len(tok)

    if short_sentence_mode == "discard":
        # Drop sentences with fewer than min_words words
        result = []
        for sl, tl in zip(sentence_lists, sentence_token_lists):
            sent_text = "".join(tl).strip()
            if len(sent_text.split()) >= min_words and sl:
                result.append(float(np.mean(sl)))
        return result if result else [float(np.mean(scores))]
    else:
        # Default: merge short sentences (< min_words words) into adjacent
        i = 0
        while i < len(sentence_lists):
            sent_text = "".join(sentence_token_lists[i]).strip()
            if len(sent_text.split()) < min_words:
                if i < len(sentence_lists) - 1:
                    sentence_lists[i + 1] = sentence_lists[i] + sentence_lists[i + 1]
                    sentence_token_lists[i + 1] = sentence_token_lists[i] + sentence_token_lists[i + 1]
                    sentence_lists.pop(i)
                    sentence_token_lists.pop(i)
                elif i > 0:
                    sentence_lists[i - 1].extend(sentence_lists[i])
                    sentence_token_lists[i - 1].extend(sentence_token_lists[i])
                    sentence_lists.pop(i)
                    sentence_token_lists.pop(i)
                else:
                    i += 1
            else:
                i += 1

        result = [float(np.mean(s)) for s in sentence_lists if s]
        return result if result else [float(np.mean(scores))]


def _split_tokens_into_sentences_with_text(
    tokens: list[str],
) -> list[tuple[str, int, int]]:
    """Split tokens into sentences, returning (text, char_start, char_end).

    Character ranges are relative to the full concatenated token text.
    Uses the same sentence boundary logic as ``_split_tokens_into_sentences``.
    """
    if not tokens:
        return []

    text = "".join(tokens)
    split_positions = sorted(
        set(m.end() for m in _SENTENCE_SPLIT_RE.finditer(text))
    )

    # Precompute token start positions
    token_starts: list[int] = []
    pos = 0
    for tok in tokens:
        token_starts.append(pos)
        pos += len(tok)

    sentence_token_groups: list[list[int]] = [[]]
    char_pos = 0
    split_idx = 0

    for i, tok in enumerate(tokens):
        tok_mid = char_pos + len(tok) / 2
        while (
            split_idx < len(split_positions)
            and tok_mid >= split_positions[split_idx]
        ):
            split_idx += 1
            sentence_token_groups.append([])
        sentence_token_groups[-1].append(i)
        char_pos += len(tok)

    # Merge short sentence groups (< 5 words) into next sentence
    i = 0
    while i < len(sentence_token_groups):
        grp = sentence_token_groups[i]
        grp_text = "".join(tokens[j] for j in grp).strip() if grp else ""
        if len(grp_text.split()) < 5:
            if i < len(sentence_token_groups) - 1:
                sentence_token_groups[i + 1] = grp + sentence_token_groups[i + 1]
                sentence_token_groups.pop(i)
            elif i > 0:
                sentence_token_groups[i - 1].extend(grp)
                sentence_token_groups.pop(i)
            else:
                i += 1
        else:
            i += 1

    result: list[tuple[str, int, int]] = []
    for tok_indices in sentence_token_groups:
        if not tok_indices:
            continue
        sent_text = "".join(tokens[i] for i in tok_indices)
        char_start = token_starts[tok_indices[0]]
        char_end = char_start + len(sent_text)
        result.append((sent_text, char_start, char_end))

    return result if result else [(text, 0, len(text))]


def _get_assistant_turns_for_rollout(rollout: dict) -> list[int]:
    """Return the bloom turn number for each assistant message in the target view."""
    events = rollout["transcript"]["events"]
    current_turn = 0
    assistant_turns: list[int] = []

    for event in events:
        if "target" not in event.get("view", []):
            continue
        msg = event.get("edit", {}).get("message", {})
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            current_turn += 1
        elif role == "assistant":
            if isinstance(content, list):
                has_content = any(
                    isinstance(b, dict) and (
                        (b.get("type") == "reasoning" and b.get("reasoning", "").strip())
                        or (b.get("type") == "text" and b.get("text", "").strip())
                    )
                    for b in content
                )
            else:
                has_content = bool(str(content).strip())
            if has_content:
                assistant_turns.append(current_turn)

    return assistant_turns


def load_per_sentence_scores(
    token_scores_path: Path,
    rollout_dir: str,
    rollouts_cache: dict[str, list[dict]],
    short_sentence_mode: str = "merge",
    min_words: int = 5,
) -> dict[int, dict[int, list[float]]] | None:
    """Compute per-sentence average scores per turn from token_scores.json.

    Returns ``{rollout_index: {turn_number: [sentence_avg_scores]}}``
    or None if token_scores.json is missing.

    Args:
        short_sentence_mode: "merge" (default) or "discard" — how to handle
            sentences with fewer than *min_words* words.
    """
    if not token_scores_path.exists():
        return None

    with open(token_scores_path) as f:
        ts_data = json.load(f)

    # Load or retrieve cached full rollout data
    rollout_dir_str = str(rollout_dir)
    if rollout_dir_str not in rollouts_cache:
        rollout_path = Path(rollout_dir) / "rollout.json"
        with open(rollout_path) as f:
            rollouts_cache[rollout_dir_str] = json.load(f)["rollouts"]
    rollouts = rollouts_cache[rollout_dir_str]

    result: dict[int, dict[int, list[float]]] = {}
    for ts_entry in ts_data["per_rollout"]:
        idx = ts_entry["rollout_index"]
        if idx >= len(rollouts):
            continue

        tokens = ts_entry["tokens"]
        scores = ts_entry["scores"]

        # Find scored segments (consecutive non-None scores)
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

        # Group segments by turn
        turn_token_lists: dict[int, list[str]] = defaultdict(list)
        turn_score_lists: dict[int, list[float]] = defaultdict(list)
        for seg_idx, (seg_s, seg_e) in enumerate(segments):
            turn = assistant_turns[seg_idx] if seg_idx < len(assistant_turns) else 0
            turn_token_lists[turn].extend(tokens[seg_s:seg_e])
            turn_score_lists[turn].extend(scores[seg_s:seg_e])

        # Split each turn's tokens into sentences
        per_turn: dict[int, list[float]] = {}
        for turn in turn_token_lists:
            per_turn[turn] = _split_tokens_into_sentences(
                turn_token_lists[turn], turn_score_lists[turn],
                short_sentence_mode=short_sentence_mode,
                min_words=min_words,
            )

        result[idx] = per_turn

    return result


def build_sentence_scores_by_key(
    sentence_scores_by_rollout: dict[int, dict[int, list[float]]],
    var_rep_map: dict[int, tuple[int, int]],
) -> dict[tuple, list[float]]:
    """Map sentence scores to ``(var, rep, turn)`` keys."""
    result: dict[tuple, list[float]] = {}
    for idx, per_turn in sentence_scores_by_rollout.items():
        if idx not in var_rep_map:
            continue
        var, rep = var_rep_map[idx]
        for turn, sent_scores in per_turn.items():
            result[(var, rep, turn)] = sent_scores
    return result


def to_sentence_max_points(
    points: list[dict],
    sentence_scores_by_key: dict[tuple, list[float]],
) -> list[dict]:
    """Create a copy of *points* with probe_score replaced by max(sentence_scores).

    Points without sentence data keep their original probe_score.
    """
    result = []
    for p in points:
        key = (p["var"], p["rep"], p["turn"])
        sent_scores = sentence_scores_by_key.get(key, [])
        new_p = dict(p)
        if sent_scores:
            new_p["probe_score"] = max(sent_scores)
            new_p["sentence_scores"] = sent_scores
        result.append(new_p)
    return result


def load_per_sentence_texts(
    token_scores_path: Path,
    rollout_dir: str,
    rollouts_cache: dict[str, list[dict]],
) -> dict[int, dict[int, list[tuple[str, int, int]]]] | None:
    """Load per-sentence text spans from token_scores.json.

    Returns ``{rollout_index: {turn_number: [(text, char_start, char_end)]}}``
    or None if token_scores.json is missing.  Character ranges are relative
    to each turn's full concatenated token text.
    """
    if not token_scores_path.exists():
        return None

    with open(token_scores_path) as f:
        ts_data = json.load(f)

    rollout_dir_str = str(rollout_dir)
    if rollout_dir_str not in rollouts_cache:
        rollout_path = Path(rollout_dir) / "rollout.json"
        with open(rollout_path) as f:
            rollouts_cache[rollout_dir_str] = json.load(f)["rollouts"]
    rollouts = rollouts_cache[rollout_dir_str]

    result: dict[int, dict[int, list[tuple[str, int, int]]]] = {}
    for ts_entry in ts_data["per_rollout"]:
        idx = ts_entry["rollout_index"]
        if idx >= len(rollouts):
            continue

        tokens = ts_entry["tokens"]
        scores = ts_entry["scores"]

        # Find scored segments (consecutive non-None scores)
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

        assistant_turns = _get_assistant_turns_for_rollout(rollouts[idx])

        turn_token_lists: dict[int, list[str]] = defaultdict(list)
        for seg_idx, (seg_s, seg_e) in enumerate(segments):
            turn = assistant_turns[seg_idx] if seg_idx < len(assistant_turns) else 0
            turn_token_lists[turn].extend(tokens[seg_s:seg_e])

        per_turn: dict[int, list[tuple[str, int, int]]] = {}
        for turn, tok_list in turn_token_lists.items():
            per_turn[turn] = _split_tokens_into_sentences_with_text(tok_list)

        result[idx] = per_turn

    return result


def build_sentence_texts_by_key(
    sentence_texts_by_rollout: dict[int, dict[int, list[tuple[str, int, int]]]],
    var_rep_map: dict[int, tuple[int, int]],
) -> dict[tuple, list[tuple[str, int, int]]]:
    """Map sentence texts to ``(var, rep, turn)`` keys."""
    result: dict[tuple, list[tuple[str, int, int]]] = {}
    for idx, per_turn in sentence_texts_by_rollout.items():
        if idx not in var_rep_map:
            continue
        var, rep = var_rep_map[idx]
        for turn, sent_texts in per_turn.items():
            result[(var, rep, turn)] = sent_texts
    return result
