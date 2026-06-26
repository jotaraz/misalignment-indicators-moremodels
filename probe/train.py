"""
Train probes on GLM-4.7 Flash using generated synthetic transcripts.

Supports two labeling modes:
  --label-mode turn  : positive = tokens in indicator turns, negative = tokens in non-indicator turns
  --label-mode span  : positive = tokens matching indicator span text, negative = all other assistant tokens

Supports two threshold tuning modes (--threshold-mode):
  --threshold-mode sentence : (default) tune threshold on per-sentence average scores;
      a sentence is TP if it overlaps with a GT positive span
  --threshold-mode turn     : tune threshold on per-token scores (legacy behavior)

Supports two indicator sets (--indicator-set):
  - finegrain (default): data from probe/data/v2_2/
  - behavior: data from probe/data/v2_2_general_behavior/

Loads transcripts, tokenizes them, extracts activations from GLM-4.7 Flash,
trains a LogisticRegressionDetector, and saves the trained probe in a format
compatible with probe_eval/evaluate.py.

Usage:
    python -m probe.train --indicator "Observer Suspicion Modeling" --label-mode turn
    python -m probe.train --all --label-mode span --reg-coeff 10.0
    python -m probe.train --indicator-set behavior --all --label-mode turn
    python -m probe.train --all --threshold-mode turn  # use legacy turn-level thresholds
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import trange

# Add project root and deception-detection to path
REPO_ROOT = Path(__file__).parent.parent
DD_ROOT = REPO_ROOT / "ood_misalignment_eval" / "deception-detection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DD_ROOT))

from deception_detection.activations import Activations
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Dialogue, Message

DATA_DIRS = {
    "finegrain": Path(__file__).parent / "data" / "v2_2",
    "behavior": Path(__file__).parent / "data" / "v2_2_general_behavior",
    "v2_3": Path(__file__).parent / "data" / "v2_3",
    "v2_4": Path(__file__).parent / "data" / "v2_4",
    "v2_5": Path(__file__).parent / "data" / "v3_v2_5",
    "v2_6": Path(__file__).parent / "data" / "v4_v2_6",
}
OUTPUT_DIRS = {
    "finegrain": Path(__file__).parent / "probes" / "finegrain",
    "behavior": Path(__file__).parent / "probes" / "behavior",
    "v2_3": Path(__file__).parent / "probes" / "v2_3",
    "v2_4": Path(__file__).parent / "probes" / "v2_4",
    "v2_5": Path(__file__).parent / "probes" / "v3_v2_5",
    "v2_6": Path(__file__).parent / "probes" / "v4_v2_6",
}
MODEL_NAME = ModelName.GLM_FLASH  # "glm-9b-flash", 47 layers
DEFAULT_DETECT_LAYERS = [26, 28, 30, 32, 34]  # 5 layers in the 26-34 range

# Zero padding to match probe_eval/evaluate.py (only score assistant tokens)
DEFAULT_PADDING = {
    "gemma": {"left": 0, "right": 0},
    "mistral": {"left": 0, "right": 0},
    "llama": {"left": 0, "right": 0},
    "qwen": {"left": 0, "right": 0},
    "glm": {"left": 0, "right": 0},
}


def indicator_name_to_filename(name: str) -> str:
    """Convert indicator name to a safe filename slug."""
    return name.lower().replace(" ", "_").replace("-", "_")


def load_transcripts(indicator_name: str, data_dir: Path) -> dict[str, Any]:
    """Load generated transcripts for an indicator."""
    filename = indicator_name_to_filename(indicator_name)
    path = data_dir / f"{filename}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No generated data found at {path}. Run `python -m probe.generate` first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-turn data structures
# ---------------------------------------------------------------------------


@dataclass
class TurnSample:
    """One training sample: a full dialogue with detection on a single assistant turn."""
    dialogue: Dialogue       # Full conversation, detect=True on exactly one turn
    is_positive: bool        # Whether this turn has the indicator
    spans: list[str] | None  # Span texts (for span-level mode), None for negative turns


def build_turn_samples(data: dict[str, Any]) -> list[TurnSample]:
    """
    Extract per-turn training samples from all transcripts.

    For each transcript, each assistant turn becomes one TurnSample.
    The full conversation is kept for context, but detect=True only on the target turn.

    Positive (deceptive) transcripts with no has_indicator=True turns are skipped,
    as they represent failed generations that would add noise to training.
    """
    samples: list[TurnSample] = []
    n_skipped = 0
    n_malformed = 0

    for transcript in data["transcripts"]:
        messages = transcript["messages"]
        if any("content" not in m or "role" not in m for m in messages):
            n_malformed += 1
            continue
        per_turn_map: dict[int, dict] = {
            tl["turn_index"]: tl for tl in transcript.get("per_turn_labels", [])
        }

        # Skip positive transcripts where no turn has the indicator
        if transcript.get("label") == "deceptive":
            has_any_indicator = any(
                tl.get("has_indicator", False)
                for tl in transcript.get("per_turn_labels", [])
            )
            if not has_any_indicator:
                n_skipped += 1
                continue

        for turn_idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Build dialogue with detect=True only on this turn
            dialogue: Dialogue = []
            for j, m in enumerate(messages):
                detect = (j == turn_idx)
                dialogue.append(Message(role=m["role"], content=m["content"], detect=detect))

            tl = per_turn_map.get(turn_idx, {"has_indicator": False})
            is_positive = bool(tl.get("has_indicator", False))
            spans = tl.get("spans") if is_positive else None

            samples.append(TurnSample(
                dialogue=dialogue,
                is_positive=is_positive,
                spans=spans,
            ))

    if n_skipped > 0:
        print(f"  Skipped {n_skipped} deceptive transcripts with no indicator turns")
    if n_malformed > 0:
        print(f"  Skipped {n_malformed} transcripts with malformed messages (missing role/content)")

    return samples


# ---------------------------------------------------------------------------
# Span-level mask narrowing
# ---------------------------------------------------------------------------


def narrow_mask_to_reasoning(
    toks: TokenizedDataset,
    tokenizer: Any,
) -> None:
    """
    Narrow detection_mask to only include reasoning tokens (inside <think>...</think>).

    For assistant messages containing <think>REASONING</think>RESPONSE, only the
    REASONING tokens are kept in the detection mask. The <think> and </think> tag
    tokens themselves are excluded.

    Modifies toks.detection_mask in-place.
    """
    if toks.detection_mask is None:
        return

    for batch_idx in range(toks.tokens.shape[0]):
        mask = toks.detection_mask[batch_idx]
        tokens = toks.tokens[batch_idx]
        attn_mask = toks.attention_mask[batch_idx].bool()

        # Decode attended tokens to strings
        attended_pos = attn_mask.nonzero(as_tuple=True)[0]
        str_tokens = [tokenizer.decode([tokens[p].item()]) for p in attended_pos]

        # Reconstruct full text and find <think>...</think> content char ranges
        full_text = "".join(str_tokens)
        reasoning_char_ranges: list[tuple[int, int]] = []
        search_start = 0
        while True:
            start = full_text.find("<think>", search_start)
            if start == -1:
                break
            content_start = start + len("<think>")
            end = full_text.find("</think>", content_start)
            if end == -1:
                # Unclosed <think>: treat rest as reasoning
                reasoning_char_ranges.append((content_start, len(full_text)))
                break
            reasoning_char_ranges.append((content_start, end))
            search_start = end + len("</think>")

        # Map token positions to reasoning ranges and narrow the mask
        char_offset = 0
        for j, tok_str in enumerate(str_tokens):
            tok_start = char_offset
            tok_end = char_offset + len(tok_str)

            in_reasoning = any(
                tok_start < r_end and tok_end > r_start
                for r_start, r_end in reasoning_char_ranges
            )

            pos = attended_pos[j]
            if mask[pos] and not in_reasoning:
                mask[pos] = False

            char_offset = tok_end


def narrow_mask_to_spans(
    toks: TokenizedDataset,
    spans: list[str],
) -> None:
    """
    Narrow detection_mask to only include tokens that overlap with span texts.

    Modifies toks.detection_mask in-place. Operates on the single dialogue at index 0
    (since we tokenize one dialogue at a time).
    """
    mask = toks.detection_mask[0]
    detected_indices = mask.nonzero(as_tuple=True)[0]
    if len(detected_indices) == 0 or not spans:
        return

    # Reconstruct text from detected tokens
    detected_str_tokens = [toks.str_tokens[0][j] for j in detected_indices]
    detected_text = "".join(detected_str_tokens)

    # Build new mask: only tokens overlapping with any span
    new_mask = torch.zeros_like(mask)

    for span_text in spans:
        # Find span in the reconstructed detected-region text
        search_start = 0
        while True:
            idx = detected_text.find(span_text, search_start)
            if idx == -1:
                break
            end = idx + len(span_text)

            # Map character range → token indices
            char_pos = 0
            for k, tok_str in enumerate(detected_str_tokens):
                tok_end = char_pos + len(tok_str)
                if char_pos < end and tok_end > idx:
                    new_mask[detected_indices[k]] = True
                char_pos = tok_end

            search_start = end  # advance past this match

    toks.detection_mask[0] = new_mask


# ---------------------------------------------------------------------------
# Sentence-level helpers for threshold tuning
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _normalize_text(text: str) -> str:
    """Normalize whitespace and case for fuzzy span matching."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def _sentence_overlaps_span(sentence_text: str, span_texts: list[str]) -> bool:
    """Check if a sentence overlaps with any GT span (with loose matching).

    A sentence is considered overlapping if, after normalizing whitespace and case:
      - the sentence is a substring of a span, OR
      - a span is a substring of the sentence, OR
      - they share ≥80% character overlap (SequenceMatcher ratio).
    """
    from difflib import SequenceMatcher

    norm_sent = _normalize_text(sentence_text)
    if not norm_sent or len(norm_sent) < 5:
        return False
    for span in span_texts:
        norm_span = _normalize_text(span)
        if not norm_span:
            continue
        # Exact or substring match
        if norm_sent in norm_span or norm_span in norm_sent:
            return True
        # Fuzzy: check longest common substring ratio
        ratio = SequenceMatcher(None, norm_sent, norm_span).ratio()
        if ratio >= 0.8:
            return True
    return False


def _split_tokens_into_sentences_with_text(
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


@dataclass
class ValSampleActivations:
    """Per-sample val data preserving token structure for sentence-level thresholds."""
    activations: torch.Tensor  # [n_detected_tokens, n_layers, n_features]
    str_tokens: list[str]      # detected token strings
    is_positive: bool
    spans: list[str] | None


def _extract_val_sample_activations(
    val_samples: list[TurnSample],
    model: Any,
    tokenizer: Any,
    detect_layers: list[int],
    reasoning_only: bool = False,
    reasoning_positive_only: bool = False,
) -> list[ValSampleActivations]:
    """Extract per-sample val activations preserving token structure.

    Unlike extract_activations() which concatenates everything, this returns
    per-sample data needed for sentence-level threshold tuning.
    Does NOT apply span narrowing so that full sentence text is available.
    """
    template_kwargs: dict[str, Any] = {"clear_thinking": False}
    results: list[ValSampleActivations] = []

    n_skipped = 0
    for i in trange(len(val_samples), desc="Extracting val sample activations"):
        sample = val_samples[i]

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=[sample.dialogue],
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                template_kwargs=template_kwargs,
            )
        except (AssertionError, Exception) as e:
            n_skipped += 1
            if n_skipped <= 3:
                print(f"  Skipping val sample {i}: {type(e).__name__}: {str(e)[:120]}")
            continue

        if reasoning_only or (reasoning_positive_only and sample.is_positive):
            narrow_mask_to_reasoning(toks, tokenizer)

        # Do NOT narrow to spans — we need full sentence text for matching

        if toks.detection_mask is not None and not toks.detection_mask.any():
            continue

        # Get detected token strings
        mask = toks.detection_mask[0]
        detected_indices = mask.nonzero(as_tuple=True)[0]
        if len(detected_indices) == 0:
            continue
        detected_str_tokens = [toks.str_tokens[0][j.item()] for j in detected_indices]

        acts = Activations.from_model(
            model, toks, batch_size=1, layers=detect_layers
        )
        masked = acts.get_masked_activations()  # [n_detected, n_layers, n_features]

        if masked.numel() == 0:
            del acts
            continue

        results.append(ValSampleActivations(
            activations=masked.cpu(),
            str_tokens=detected_str_tokens,
            is_positive=sample.is_positive,
            spans=sample.spans,
        ))
        del acts

    if n_skipped:
        print(f"  Warning: skipped {n_skipped}/{len(val_samples)} val samples due to tokenization errors")

    return results


def _sweep_thresholds(
    scores_arr: np.ndarray,
    labels_arr: np.ndarray,
) -> dict[str, Any]:
    """Sweep thresholds on (scores, labels) and return best-accuracy, best-F1,
    fixed-FPR operating points, AUROC, and PR-AUC.

    Returns a dict with keys: auroc, pr_auc, best_accuracy, threshold_accuracy,
    precision_accuracy, recall_accuracy, best_f1, threshold_f1, precision_f1,
    recall_f1, fpr_1pct, fpr_5pct, n_pos, n_neg.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    n_total = len(labels_arr)
    n_pos_total = int(labels_arr.sum())
    n_neg_total = n_total - n_pos_total

    empty: dict[str, Any] = {
        "auroc": None, "pr_auc": None,
        "best_accuracy": 0.0, "threshold_accuracy": float("inf"),
        "precision_accuracy": None, "recall_accuracy": None,
        "best_f1": 0.0, "threshold_f1": float("inf"),
        "precision_f1": None, "recall_f1": None,
        "fpr_1pct": {"threshold": None, "tpr": None, "fpr": None},
        "fpr_5pct": {"threshold": None, "tpr": None, "fpr": None},
        "n_pos": n_pos_total, "n_neg": n_neg_total,
    }
    if n_total == 0:
        return empty

    # Threshold-free ranking metrics
    if n_pos_total > 0 and n_neg_total > 0:
        auroc = float(roc_auc_score(labels_arr, scores_arr))
        pr_auc = float(average_precision_score(labels_arr, scores_arr))
    else:
        auroc = None
        pr_auc = None

    # Build threshold candidates
    unique_scores = sorted(set(scores_arr.tolist()))
    thresholds = [unique_scores[0] - 1.0]
    if len(unique_scores) > 1:
        thresholds += [
            (unique_scores[i] + unique_scores[i + 1]) / 2
            for i in range(len(unique_scores) - 1)
        ]
    thresholds.append(unique_scores[-1] + 1.0)

    # Sweep
    best_acc, best_acc_t = -1.0, float("inf")
    best_f1, best_f1_t = -1.0, float("inf")
    best_acc_tp = best_acc_fp = best_acc_fn = 0
    best_f1_tp = best_f1_fp = best_f1_fn = 0

    fpr_targets = {"fpr_1pct": 0.01, "fpr_5pct": 0.05}
    fpr_results: dict[str, dict[str, Any]] = {
        k: {"threshold": None, "tpr": None, "fpr": None} for k in fpr_targets
    }

    for t in thresholds:
        preds = (scores_arr > t).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())
        tn = int(((preds == 0) & (labels_arr == 0)).sum())

        acc = (tp + tn) / n_total
        if acc > best_acc or (acc == best_acc and t > best_acc_t):
            best_acc = acc
            best_acc_t = t
            best_acc_tp, best_acc_fp, best_acc_fn = tp, fp, fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1 or (f1 == best_f1 and f1 == 0.0 and t > best_f1_t):
            best_f1 = f1
            best_f1_t = t
            best_f1_tp, best_f1_fp, best_f1_fn = tp, fp, fn

        fpr = fp / n_neg_total if n_neg_total > 0 else 0.0
        tpr = tp / n_pos_total if n_pos_total > 0 else 0.0
        for key, target_fpr in fpr_targets.items():
            if fpr <= target_fpr:
                if fpr_results[key]["threshold"] is None or t < fpr_results[key]["threshold"]:
                    fpr_results[key] = {
                        "threshold": float(t),
                        "tpr": float(tpr),
                        "fpr": float(fpr),
                    }

    def _pr(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
        p = tp / (tp + fp) if (tp + fp) > 0 else None
        r = tp / (tp + fn) if (tp + fn) > 0 else None
        return p, r

    p_acc, r_acc = _pr(best_acc_tp, best_acc_fp, best_acc_fn)
    p_f1, r_f1 = _pr(best_f1_tp, best_f1_fp, best_f1_fn)

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_accuracy": best_acc,
        "threshold_accuracy": best_acc_t,
        "precision_accuracy": p_acc,
        "recall_accuracy": r_acc,
        "best_f1": best_f1,
        "threshold_f1": best_f1_t,
        "precision_f1": p_f1,
        "recall_f1": r_f1,
        **fpr_results,
        "n_pos": n_pos_total,
        "n_neg": n_neg_total,
    }


def _compute_val_thresholds_sentence(
    detector: LogisticRegressionDetector,
    val_sample_acts: list[ValSampleActivations],
    layer_idx: int,
) -> dict[str, Any]:
    """Compute optimal thresholds using sentence-level scores on val data.

    Computes three sets of metrics:
      1. **span_overlap** (strict): a sentence is TP only if its text overlaps
         a GT span.
      2. **same_turn**: a sentence is TP if it is in the same assistant turn as
         any GT span (i.e. the turn is positive), regardless of text overlap.
      3. **turn_level**: scores are aggregated per turn (max sentence score);
         a turn is TP if it contains any GT span.

    For each set, computes AUROC, PR-AUC, best-F1 threshold, best-accuracy
    threshold, and fixed-FPR operating points (1%, 5%).

    Args:
        detector: Trained single-layer detector.
        val_sample_acts: Per-sample val activations with token structure.
        layer_idx: Index into the layer dimension.

    Returns dict with sub-dicts "span_overlap", "same_turn", "turn_level",
    plus top-level backward-compat keys mirroring "span_overlap".
    """
    # Collect per-sentence scores & labels (span-overlap, same-turn, clean-label)
    span_scores: list[float] = []
    span_labels: list[int] = []
    same_turn_scores: list[float] = []
    same_turn_labels: list[int] = []
    clean_scores: list[float] = []
    clean_labels: list[int] = []
    # Collect per-turn scores & labels
    turn_scores: list[float] = []
    turn_labels: list[int] = []

    for sample_acts in val_sample_acts:
        layer_acts = sample_acts.activations[:, layer_idx:layer_idx+1, :]
        token_scores = detector.get_score_tensor(layer_acts).detach().cpu().numpy()

        sentences = _split_tokens_into_sentences_with_text(
            sample_acts.str_tokens, token_scores.tolist(),
        )

        is_positive_turn = bool(sample_acts.is_positive and sample_acts.spans)

        # Per-sentence metrics
        max_sent_score = float("-inf")
        for mean_score, sent_text in sentences:
            # Span-overlap label
            if is_positive_turn:
                span_pos = _sentence_overlaps_span(sent_text, sample_acts.spans)
            else:
                span_pos = False
            span_scores.append(mean_score)
            span_labels.append(int(span_pos))

            # Same-turn label: positive if the turn has any GT span
            same_turn_scores.append(mean_score)
            same_turn_labels.append(int(is_positive_turn))

            # Clean-label: pos = overlaps span, neg = not in positive turn
            # Sentences in positive turns that don't overlap spans are excluded
            if span_pos:
                clean_scores.append(mean_score)
                clean_labels.append(1)
            elif not is_positive_turn:
                clean_scores.append(mean_score)
                clean_labels.append(0)
            # else: same-turn non-overlapping → excluded

            if mean_score > max_sent_score:
                max_sent_score = mean_score

        # Turn-level: one entry per sample (max sentence score)
        if sentences:
            turn_scores.append(max_sent_score)
            turn_labels.append(int(is_positive_turn))

    # Compute metrics for each granularity
    empty_sweep: dict[str, Any] = {
        "auroc": None, "pr_auc": None,
        "best_accuracy": 0.0, "threshold_accuracy": float("inf"),
        "precision_accuracy": None, "recall_accuracy": None,
        "best_f1": 0.0, "threshold_f1": float("inf"),
        "precision_f1": None, "recall_f1": None,
        "fpr_1pct": {"threshold": None, "tpr": None, "fpr": None},
        "fpr_5pct": {"threshold": None, "tpr": None, "fpr": None},
        "n_pos": 0, "n_neg": 0,
    }

    if span_scores:
        span_metrics = _sweep_thresholds(np.array(span_scores), np.array(span_labels))
    else:
        span_metrics = empty_sweep

    if same_turn_scores:
        same_turn_metrics = _sweep_thresholds(np.array(same_turn_scores), np.array(same_turn_labels))
    else:
        same_turn_metrics = empty_sweep

    if turn_scores:
        turn_metrics = _sweep_thresholds(np.array(turn_scores), np.array(turn_labels))
    else:
        turn_metrics = empty_sweep

    if clean_scores:
        clean_metrics = _sweep_thresholds(np.array(clean_scores), np.array(clean_labels))
    else:
        clean_metrics = empty_sweep

    # Build result with sub-dicts and backward-compat top-level keys
    result: dict[str, Any] = {
        "threshold_mode": "sentence",
        "span_overlap": span_metrics,
        "same_turn": same_turn_metrics,
        "turn_level": turn_metrics,
        "clean_label": clean_metrics,
        # Backward-compat top-level keys (mirror span_overlap)
        "auroc": span_metrics["auroc"],
        "pr_auc": span_metrics["pr_auc"],
        "best_accuracy": span_metrics["best_accuracy"],
        "threshold_accuracy": span_metrics["threshold_accuracy"],
        "precision_accuracy": span_metrics["precision_accuracy"],
        "recall_accuracy": span_metrics["recall_accuracy"],
        "best_f1": span_metrics["best_f1"],
        "threshold_f1": span_metrics["threshold_f1"],
        "precision_f1": span_metrics["precision_f1"],
        "recall_f1": span_metrics["recall_f1"],
        "fpr_1pct": span_metrics["fpr_1pct"],
        "fpr_5pct": span_metrics["fpr_5pct"],
        "n_pos_sentences": span_metrics["n_pos"],
        "n_neg_sentences": span_metrics["n_neg"],
    }
    return result


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def extract_activations(
    samples: list[TurnSample],
    label_mode: str,
    model: Any,
    tokenizer: Any,
    detect_layers: list[int],
    reasoning_only: bool = False,
    reasoning_positive_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process samples one at a time and accumulate masked activations.

    Args:
        samples: List of TurnSample objects
        label_mode: "turn" or "span"
        model: Loaded GLM model
        tokenizer: Loaded GLM tokenizer
        detect_layers: Layers to extract
        reasoning_only: If True, narrow BOTH positive and negative samples to
            reasoning tokens (inside <think>...</think>).
        reasoning_positive_only: If True, narrow ONLY positive samples to
            reasoning tokens; negatives keep all assistant tokens (reasoning +
            response). Broader negative pool → sharper contrast. Mutually
            exclusive with reasoning_only.

    Returns:
        (positive_acts, negative_acts) each of shape [n_tokens, n_layers, n_features]
    """
    template_kwargs: dict[str, Any] = {"clear_thinking": False}
    pos_acts_list: list[torch.Tensor] = []
    neg_acts_list: list[torch.Tensor] = []

    n_skipped = 0
    for i in trange(len(samples), desc="Extracting activations"):
        sample = samples[i]

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=[sample.dialogue],
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                template_kwargs=template_kwargs,
            )
        except (AssertionError, Exception) as e:
            n_skipped += 1
            if n_skipped <= 3:
                print(f"  Skipping sample {i}: {type(e).__name__}: {str(e)[:120]}")
            continue

        # Narrow detection mask to reasoning tokens.
        # - reasoning_only: narrow all samples
        # - reasoning_positive_only: narrow only positive samples (negatives
        #   keep full assistant tokens → richer negative pool)
        if reasoning_only or (reasoning_positive_only and sample.is_positive):
            narrow_mask_to_reasoning(toks, tokenizer)

        # For span mode on positive turns: narrow detection mask to span tokens
        if label_mode == "span" and sample.is_positive and sample.spans:
            narrow_mask_to_spans(toks, sample.spans)

        # Skip if detection mask is empty after narrowing
        if toks.detection_mask is not None and not toks.detection_mask.any():
            continue

        acts = Activations.from_model(
            model, toks, batch_size=1, layers=detect_layers
        )
        masked = acts.get_masked_activations()  # [n_detected_tokens, n_layers, n_features]

        if masked.numel() == 0:
            del acts
            continue

        if sample.is_positive:
            pos_acts_list.append(masked.cpu())
        else:
            neg_acts_list.append(masked.cpu())

        del acts

    if n_skipped:
        print(f"  Skipped {n_skipped} samples due to tokenization errors")

    if not pos_acts_list or not neg_acts_list:
        raise ValueError("No positive or negative activations extracted. Check your data.")

    return torch.cat(pos_acts_list, dim=0), torch.cat(neg_acts_list, dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_detector(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    detect_layers: list[int],
    reg_coeff: float = 10.0,
    normalize: bool = True,
    lr: float = 1e-2,
    n_steps: int = 500,
) -> LogisticRegressionDetector:
    """
    Train a LogisticRegressionDetector from pre-extracted activation tensors.

    Uses PyTorch L2-regularized logistic regression on GPU for speed.

    Args:
        pos_acts: [n_pos_tokens, n_layers, n_features]
        neg_acts: [n_neg_tokens, n_layers, n_features]
        detect_layers: Layer indices used
        reg_coeff: Regularization coefficient (higher = more regularization)
        normalize: Whether to standardize features
        lr: Learning rate for Adam optimizer
        n_steps: Number of optimization steps

    Returns:
        Trained LogisticRegressionDetector
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos, n_layers, n_features = pos_acts.shape
    n_neg = neg_acts.shape[0]
    print(f"  LR training: {n_pos} positive tokens, {n_neg} negative tokens, "
          f"{n_layers} layers, {n_features} features")

    # Flatten layers into features: [n_tokens, n_layers * n_features]
    X = torch.cat([
        pos_acts.reshape(n_pos, n_layers * n_features),
        neg_acts.reshape(n_neg, n_layers * n_features),
    ], dim=0).float()
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).float()

    # Normalize
    scaler_mean = None
    scaler_scale = None
    if normalize:
        scaler_mean = X.mean(dim=0)
        scaler_scale = X.std(dim=0).clamp(min=1e-8)
        X = (X - scaler_mean) / scaler_scale
        scaler_mean = scaler_mean.reshape(n_layers, n_features)
        scaler_scale = scaler_scale.reshape(n_layers, n_features)

    X = X.to(device)
    y = y.to(device)

    # Train logistic regression (no bias, matching sklearn fit_intercept=False)
    linear = nn.Linear(n_layers * n_features, 1, bias=False).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(n_steps):
        logits = linear(X).squeeze(-1)
        loss = loss_fn(logits, y)
        # L2 regularization: sklearn uses C = 1/reg_coeff, equivalent to
        # adding reg_coeff * ||w||^2 / (2 * n_samples) penalty
        l2 = reg_coeff * linear.weight.square().sum() / (2 * len(y))
        (loss + l2).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Pack into detector
    detector = LogisticRegressionDetector(
        layers=detect_layers, reg_coeff=reg_coeff, normalize=normalize
    )
    detector.directions = linear.weight.detach().cpu().reshape(n_layers, n_features)
    detector.scaler_mean = scaler_mean
    detector.scaler_scale = scaler_scale

    return detector


class BilinearDetector:
    """Bilinear probe: score = sum_r (w1_r · x)(w2_r · x) [+ w_linear · x].

    Stores normalization info and supports save/load for evaluation.
    """

    def __init__(self, layers: list[int], rank: int = 2, use_linear: bool = True,
                 reg_coeff: float = 1.0, normalize: bool = True):
        self.layers = layers
        self.rank = rank
        self.use_linear = use_linear
        self.reg_coeff = reg_coeff
        self.normalize = normalize
        # Weights: set after training
        self.w1: torch.Tensor | None = None  # [rank, n_layers * n_features]
        self.w2: torch.Tensor | None = None  # [rank, n_layers * n_features]
        self.w_linear: torch.Tensor | None = None  # [1, n_layers * n_features] or None
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_scale: torch.Tensor | None = None

    def get_score_tensor(self, acts: torch.Tensor) -> torch.Tensor:
        """Score activations. acts: [n_tokens, n_layers, n_features]."""
        n_tokens, n_layers, n_features = acts.shape
        x = acts.reshape(n_tokens, n_layers * n_features).float()
        if self.normalize and self.scaler_mean is not None:
            sm = self.scaler_mean.reshape(-1)
            ss = self.scaler_scale.reshape(-1)
            x = (x - sm) / ss
        # Bilinear: sum_r (w1_r · x) * (w2_r · x)
        proj1 = x @ self.w1.T  # [n_tokens, rank]
        proj2 = x @ self.w2.T  # [n_tokens, rank]
        score = (proj1 * proj2).sum(dim=-1)  # [n_tokens]
        if self.use_linear and self.w_linear is not None:
            score = score + (x @ self.w_linear.T).squeeze(-1)
        return score

    def save(self, path):
        import pickle
        data = {
            "type": "bilinear",
            "layers": self.layers,
            "rank": self.rank,
            "use_linear": self.use_linear,
            "w1": self.w1,
            "w2": self.w2,
            "w_linear": self.w_linear,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "normalize": self.normalize,
            "reg_coeff": self.reg_coeff,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path) -> "BilinearDetector":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(
            layers=data["layers"], rank=data["rank"],
            use_linear=data["use_linear"],
            reg_coeff=data["reg_coeff"], normalize=data["normalize"],
        )
        det.w1 = data["w1"]
        det.w2 = data["w2"]
        det.w_linear = data["w_linear"]
        det.scaler_mean = data["scaler_mean"]
        det.scaler_scale = data["scaler_scale"]
        return det


def train_bilinear_detector(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    detect_layers: list[int],
    rank: int = 2,
    use_linear: bool = True,
    reg_coeff: float = 1.0,
    normalize: bool = True,
    lr: float = 1e-3,
    n_steps: int = 1000,
) -> BilinearDetector:
    """Train a BilinearDetector from pre-extracted activation tensors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos, n_layers, n_features = pos_acts.shape
    n_neg = neg_acts.shape[0]
    d = n_layers * n_features
    print(f"  Bilinear training: {n_pos} pos, {n_neg} neg, d={d}, rank={rank}, linear={use_linear}")

    X = torch.cat([
        pos_acts.reshape(n_pos, d),
        neg_acts.reshape(n_neg, d),
    ], dim=0).float()
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).float()

    scaler_mean = None
    scaler_scale = None
    if normalize:
        scaler_mean = X.mean(dim=0)
        scaler_scale = X.std(dim=0).clamp(min=1e-8)
        X = (X - scaler_mean) / scaler_scale

    X = X.to(device)
    y = y.to(device)

    w1 = nn.Linear(d, rank, bias=False).to(device)
    w2 = nn.Linear(d, rank, bias=False).to(device)
    w_lin = nn.Linear(d, 1, bias=False).to(device) if use_linear else None

    params = list(w1.parameters()) + list(w2.parameters())
    if w_lin is not None:
        params += list(w_lin.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(n_steps):
        proj1 = w1(X)  # [batch, rank]
        proj2 = w2(X)  # [batch, rank]
        logits = (proj1 * proj2).sum(dim=-1)
        if w_lin is not None:
            logits = logits + w_lin(X).squeeze(-1)
        loss = loss_fn(logits, y)
        l2 = reg_coeff * sum(p.square().sum() for p in params) / (2 * len(y))
        (loss + l2).backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                acc = ((logits > 0).float() == y).float().mean()
                print(f"    step {step+1}: loss={loss.item():.4f} acc={acc.item():.3f}")

    det = BilinearDetector(
        layers=detect_layers, rank=rank, use_linear=use_linear,
        reg_coeff=reg_coeff, normalize=normalize,
    )
    det.w1 = w1.weight.detach().cpu()
    det.w2 = w2.weight.detach().cpu()
    det.w_linear = w_lin.weight.detach().cpu() if w_lin is not None else None
    if normalize:
        det.scaler_mean = scaler_mean.reshape(n_layers, n_features).cpu()
        det.scaler_scale = scaler_scale.reshape(n_layers, n_features).cpu()

    return det


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _split_transcripts_train_val(
    data: dict[str, Any],
    val_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split transcript data into train and val at the transcript level.

    Uses a deterministic positional split (no shuffling): the last
    val_fraction of each class becomes val.  Data files are expected to
    be pre-shuffled on disk so that different training runs on the same
    data always use identical train/val partitions.

    Stratifies by label (deceptive/honest) to maintain class balance.
    Returns (train_data, val_data) dicts with the same structure as *data*.
    """
    transcripts = data["transcripts"]
    deceptive = [t for t in transcripts if t.get("label") == "deceptive"]
    honest = [t for t in transcripts if t.get("label") != "deceptive"]

    n_val_d = max(1, int(len(deceptive) * val_fraction))
    n_val_h = max(1, int(len(honest) * val_fraction))

    # Take val from the tail so train is always the same prefix
    train_transcripts = deceptive[:-n_val_d] + honest[:-n_val_h]
    val_transcripts = deceptive[-n_val_d:] + honest[-n_val_h:]

    base = {k: v for k, v in data.items() if k != "transcripts"}
    return (
        {**base, "transcripts": train_transcripts},
        {**base, "transcripts": val_transcripts},
    )


def _compute_val_thresholds_turn(
    detector: LogisticRegressionDetector,
    val_pos_acts: torch.Tensor,
    val_neg_acts: torch.Tensor,
    layer_idx: int,
) -> dict[str, Any]:
    """Compute optimal thresholds on held-out val activations (turn/token level).

    Args:
        detector: Trained detector (single layer).
        val_pos_acts: [n_pos_tokens, n_layers, n_features] full multi-layer acts.
        val_neg_acts: [n_neg_tokens, n_layers, n_features] full multi-layer acts.
        layer_idx: Index into the layer dimension to extract.

    Returns dict with best_accuracy, threshold_accuracy, best_f1, threshold_f1,
        precision_*, recall_*, n_pos, n_neg.
    """
    layer_pos = val_pos_acts[:, layer_idx:layer_idx+1, :]
    layer_neg = val_neg_acts[:, layer_idx:layer_idx+1, :]

    pos_scores = detector.get_score_tensor(layer_pos).detach().cpu().numpy()
    neg_scores = detector.get_score_tensor(layer_neg).detach().cpu().numpy()

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    unique_scores = sorted(set(all_scores.tolist()))
    thresholds = [unique_scores[0] - 1.0]
    if len(unique_scores) > 1:
        thresholds += [
            (unique_scores[i] + unique_scores[i + 1]) / 2
            for i in range(len(unique_scores) - 1)
        ]
    thresholds.append(unique_scores[-1] + 1.0)

    best_acc, best_acc_t = -1.0, float("inf")
    best_f1, best_f1_t = -1.0, float("inf")
    best_acc_tp = best_acc_fp = best_acc_fn = 0
    best_f1_tp = best_f1_fp = best_f1_fn = 0

    n_total = len(all_labels)
    n_pos_total = int(all_labels.sum())

    for t in thresholds:
        preds = (all_scores > t).astype(int)
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        tn = int(((preds == 0) & (all_labels == 0)).sum())

        acc = (tp + tn) / n_total
        if acc > best_acc or (acc == best_acc and t > best_acc_t):
            best_acc = acc
            best_acc_t = t
            best_acc_tp, best_acc_fp, best_acc_fn = tp, fp, fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1 or (f1 == best_f1 and f1 == 0.0 and t > best_f1_t):
            best_f1 = f1
            best_f1_t = t
            best_f1_tp, best_f1_fp, best_f1_fn = tp, fp, fn

    def _pr(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else None
        r = tp / (tp + fn) if (tp + fn) > 0 else None
        return p, r

    p_acc, r_acc = _pr(best_acc_tp, best_acc_fp, best_acc_fn)
    p_f1, r_f1 = _pr(best_f1_tp, best_f1_fp, best_f1_fn)

    return {
        "best_accuracy": best_acc,
        "threshold_accuracy": best_acc_t,
        "precision_accuracy": p_acc,
        "recall_accuracy": r_acc,
        "best_f1": best_f1,
        "threshold_f1": best_f1_t,
        "precision_f1": p_f1,
        "recall_f1": r_f1,
        "n_pos_tokens": n_pos_total,
        "n_neg_tokens": n_total - n_pos_total,
    }


def _print_sweep_metrics(m: dict[str, Any], label: str, indent: str = "      ") -> None:
    """Print metrics from a _sweep_thresholds result dict."""
    auroc_s = f"{m['auroc']:.4f}" if m['auroc'] is not None else "N/A"
    pr_auc_s = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
    print(f"{indent}AUROC={auroc_s}  PR-AUC={pr_auc_s}  "
          f"(n_pos={m['n_pos']}, n_neg={m['n_neg']})")
    print(f"{indent}Acc={m['best_accuracy']:.3f} (t={m['threshold_accuracy']:.4f})")
    p_f1 = m['precision_f1']
    r_f1 = m['recall_f1']
    p_s = f"{p_f1:.3f}" if p_f1 is not None else "N/A"
    r_s = f"{r_f1:.3f}" if r_f1 is not None else "N/A"
    print(f"{indent}F1={m['best_f1']:.3f} (t={m['threshold_f1']:.4f}, P={p_s}, R={r_s})")
    for fpr_key, fpr_label in [("fpr_1pct", "1%"), ("fpr_5pct", "5%")]:
        fp_data = m.get(fpr_key, {})
        if fp_data and fp_data.get("threshold") is not None:
            print(f"{indent}@FPR<={fpr_label}: t={fp_data['threshold']:.4f}, "
                  f"TPR={fp_data['tpr']:.3f}, FPR={fp_data['fpr']:.3f}")
        else:
            print(f"{indent}@FPR<={fpr_label}: no feasible threshold")


def _print_sentence_val_metrics(vt: dict[str, Any]) -> None:
    """Print all sentence-mode val metrics (span_overlap, same_turn, turn_level, clean_label)."""
    for key, title in [
        ("span_overlap", "Span-overlap (strict)"),
        ("same_turn", "Same-turn (relaxed)"),
        ("turn_level", "Turn-level (max sent score)"),
        ("clean_label", "Clean-label (span+ / other-turn-)"),
    ]:
        sub = vt.get(key)
        if sub is None:
            continue
        print(f"    [{title}]")
        _print_sweep_metrics(sub, title)


def train_probe(
    data: dict[str, Any],
    label_mode: str,
    detect_layers: list[int],
    model: Any,
    tokenizer: Any,
    reg_coeff: float = 10.0,
    normalize: bool = True,
    reasoning_only: bool = False,
    reasoning_positive_only: bool = False,
    val_fraction: float = 0.0,
    threshold_mode: str = "sentence",
    probe_type: str = "linear",
    bilinear_rank: int = 2,
    bilinear_use_linear: bool = True,
) -> tuple[dict[int, LogisticRegressionDetector | BilinearDetector], dict[int, dict[str, Any]], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """
    End-to-end probe training pipeline.

    1. Build per-turn samples from generated data
    2. Extract activations for all layers in one forward pass
    3. Train a separate detector per layer (linear or bilinear)
    4. If val_fraction > 0, compute optimal thresholds on held-out val data

    Args:
        data: Loaded transcript data dict
        label_mode: "turn" or "span"
        detect_layers: Layers to extract
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize
        reasoning_only: If True, only use reasoning tokens (inside <think>...</think>)
        val_fraction: Fraction of transcripts to hold out for threshold tuning.
        probe_type: "linear" or "bilinear"
        bilinear_rank: Number of bilinear interaction terms
        bilinear_use_linear: Whether to include a linear term in bilinear probe
            Split is stratified by label at the transcript level. 0.0 = no split.
        threshold_mode: "sentence" (default) tunes thresholds on per-sentence average
            scores where a sentence is TP if it overlaps a GT span. "turn" uses the
            legacy per-token threshold tuning.

    Returns:
        (detectors, val_thresholds, val_acts_per_layer) where:
        - detectors maps layer → detector
        - val_thresholds maps layer → threshold metrics dict (empty if val_fraction=0)
        - val_acts_per_layer maps layer → (val_pos_acts, val_neg_acts) tensors
    """
    # 1. Optionally split data
    if val_fraction > 0:
        train_data, val_data = _split_transcripts_train_val(data, val_fraction)
        n_train_transcripts = len(train_data["transcripts"])
        n_val_transcripts = len(val_data["transcripts"])
        print(f"Train/val split: {n_train_transcripts} train, {n_val_transcripts} val transcripts "
              f"(val_fraction={val_fraction})")
    else:
        train_data = data
        val_data = None

    # 2. Build per-turn samples
    samples = build_turn_samples(train_data)
    n_pos = sum(1 for s in samples if s.is_positive)
    n_neg = sum(1 for s in samples if not s.is_positive)
    print(f"Built {n_pos} positive + {n_neg} negative turn samples (label_mode={label_mode})")

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both positive and negative samples, got {n_pos} pos / {n_neg} neg")

    val_samples = None
    if val_data is not None:
        val_samples = build_turn_samples(val_data)
        val_n_pos = sum(1 for s in val_samples if s.is_positive)
        val_n_neg = sum(1 for s in val_samples if not s.is_positive)
        print(f"Val set: {val_n_pos} positive + {val_n_neg} negative turn samples")

    # 3. Extract activations (all layers in one forward pass per sample)
    pos_acts, neg_acts = extract_activations(
        samples, label_mode, model, tokenizer, detect_layers,
        reasoning_only=reasoning_only,
        reasoning_positive_only=reasoning_positive_only,
    )

    # Extract val activations
    val_pos_acts = val_neg_acts = None
    val_sample_acts: list[ValSampleActivations] | None = None
    if val_samples:
        if threshold_mode == "sentence":
            # Sentence-level: extract per-sample data preserving token structure
            print("Extracting val sample activations (sentence-level threshold mode)...")
            val_sample_acts = _extract_val_sample_activations(
                val_samples, model, tokenizer, detect_layers,
                reasoning_only=reasoning_only,
                reasoning_positive_only=reasoning_positive_only,
            )
            # Also extract concatenated pos/neg acts for saving to val_acts.pt
            print("Extracting val activations (for saving)...")
            val_pos_acts, val_neg_acts = extract_activations(
                val_samples, label_mode, model, tokenizer, detect_layers,
                reasoning_only=reasoning_only,
                reasoning_positive_only=reasoning_positive_only,
            )
        else:
            # Turn-level: only need concatenated pos/neg acts
            print("Extracting val activations...")
            val_pos_acts, val_neg_acts = extract_activations(
                val_samples, label_mode, model, tokenizer, detect_layers,
                reasoning_only=reasoning_only,
                reasoning_positive_only=reasoning_positive_only,
            )

    # 5. Train a separate detector per layer + compute val thresholds
    detectors: dict[int, LogisticRegressionDetector | BilinearDetector] = {}
    val_thresholds: dict[int, dict[str, Any]] = {}
    # Per-layer val activations for saving (used by clean_probes.py to recompute thresholds)
    val_acts_per_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, layer in enumerate(detect_layers):
        print(f"Training {probe_type} detector for layer {layer}...")
        layer_pos = pos_acts[:, i:i+1, :]  # [n_tokens, 1, n_features]
        layer_neg = neg_acts[:, i:i+1, :]
        if probe_type == "bilinear":
            detectors[layer] = train_bilinear_detector(
                layer_pos, layer_neg, [layer],
                rank=bilinear_rank, use_linear=bilinear_use_linear,
                reg_coeff=reg_coeff, normalize=normalize,
            )
        else:
            detectors[layer] = train_detector(
                layer_pos, layer_neg, [layer], reg_coeff, normalize
            )

        if val_samples and val_fraction > 0:
            print(f"  Computing val thresholds for layer {layer} (mode={threshold_mode})...")
            if threshold_mode == "sentence" and val_sample_acts is not None:
                val_thresholds[layer] = _compute_val_thresholds_sentence(
                    detectors[layer], val_sample_acts, i,
                )
            elif val_pos_acts is not None and val_neg_acts is not None:
                val_thresholds[layer] = _compute_val_thresholds_turn(
                    detectors[layer], val_pos_acts, val_neg_acts, i,
                )

            if layer in val_thresholds:
                vt = val_thresholds[layer]
                if threshold_mode == "sentence":
                    _print_sentence_val_metrics(vt)
                else:
                    n_label = f"n_pos_tokens={vt.get('n_pos_tokens', '?')}"
                    print(f"    Val accuracy={vt['best_accuracy']:.3f} (t={vt['threshold_accuracy']:.4f})")
                    p_f1 = vt['precision_f1']
                    r_f1 = vt['recall_f1']
                    p_s = f"{p_f1:.3f}" if p_f1 is not None else "N/A"
                    r_s = f"{r_f1:.3f}" if r_f1 is not None else "N/A"
                    print(f"    Val F1={vt['best_f1']:.3f} (t={vt['threshold_f1']:.4f}, "
                          f"P={p_s}, R={r_s})  ({n_label})")

            # Store single-layer val acts for saving
            if val_pos_acts is not None and val_neg_acts is not None:
                val_acts_per_layer[layer] = (
                    val_pos_acts[:, i:i+1, :].clone(),
                    val_neg_acts[:, i:i+1, :].clone(),
                )

    return detectors, val_thresholds, val_acts_per_layer


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_probe(
    detector: LogisticRegressionDetector,
    indicator_name: str,
    label_mode: str,
    detect_layers: list[int],
    reg_coeff: float,
    normalize: bool,
    training_data_path: str,
    n_positive_turns: int,
    n_negative_turns: int,
    probe_id: str = "",
    timestamp: str = "",
    reasoning_only: bool = False,
    reasoning_positive_only: bool = False,
    output_dir: Path | None = None,
    val_thresholds: dict[str, Any] | None = None,
    val_acts: tuple[torch.Tensor, torch.Tensor] | None = None,
    probe_type: str = "linear",
    bilinear_rank: int | None = None,
) -> Path:
    """
    Save the trained probe in probe_eval/evaluate.py compatible format.

    Folder structure:
      {output_dir}/{indicator_slug}/{label_mode}/layer{N}/
        cfg.yaml, detector.pt, training_meta.json[, val_acts.pt]
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    indicator_slug = indicator_name_to_filename(indicator_name)
    layer_str = f"layer{detect_layers[0]}" if len(detect_layers) == 1 else f"layers{'_'.join(map(str, detect_layers))}"

    base_dir = output_dir if output_dir is not None else OUTPUT_DIRS["finegrain"]
    probe_folder = base_dir / indicator_slug / label_mode / layer_str
    probe_folder.mkdir(parents=True, exist_ok=True)

    # Save detector
    detector.save(probe_folder / "detector.pt")
    print(f"Saved detector to {probe_folder / 'detector.pt'}")

    # Save cfg.yaml compatible with ExperimentConfig.from_path.
    # "repe_honesty__you_are_fact_sys" is a placeholder — probe_eval only loads the detector.
    cfg_dict = {
        "method": "bilinear" if probe_type == "bilinear" else "lr",
        "model_name": MODEL_NAME.value,
        "train_data": "repe_honesty__you_are_fact_sys",
        "eval_data": [],
        "control_data": [],
        "trim_reasoning": reasoning_only,
        "train_on_policy": False,
        "eval_on_policy": True,
        "control_on_policy": False,
        "detect_only_start_of_turn": False,
        "detect_only_last_token": False,
        "val_fraction": 0.0,
        "control_dataset_size": 0,
        "detect_layers": detect_layers,
        "detect_num_latents": None,
        "use_local_sae_acts": False,
        "use_goodfire_sae_acts": False,
        "sae_latent_whitelist": None,
        "pw_locked": False,
        "lora_path": None,
        "reg_coeff": reg_coeff,
        "normalize_acts": normalize,
        "max_llama_token_length": None,
        "use_followup_question": False,
        "id": probe_id or indicator_slug,
        "timestamp": timestamp,
    }
    with open(probe_folder / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, indent=2)
    print(f"Saved config to {probe_folder / 'cfg.yaml'}")

    # Training metadata
    meta = {
        "indicator_name": indicator_name,
        "probe_type": probe_type,
        "label_mode": label_mode,
        "reasoning_only": reasoning_only,
        "reasoning_positive_only": reasoning_positive_only,
        "training_data_path": training_data_path,
        "n_positive_turns": n_positive_turns,
        "n_negative_turns": n_negative_turns,
        "detect_layers": detect_layers,
        "reg_coeff": reg_coeff,
        "normalize": normalize,
        "model_name": MODEL_NAME.value,
        "timestamp": timestamp,
    }
    if bilinear_rank is not None:
        meta["bilinear_rank"] = bilinear_rank
    if val_thresholds:
        meta["val_thresholds"] = val_thresholds
    with open(probe_folder / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save val activations for threshold recomputation after PCA cleaning
    if val_acts is not None:
        val_pos, val_neg = val_acts
        torch.save({"pos": val_pos, "neg": val_neg}, probe_folder / "val_acts.pt")
        print(f"Saved val activations to {probe_folder / 'val_acts.pt'} "
              f"(pos={val_pos.shape}, neg={val_neg.shape})")

    return probe_folder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train probes from synthetic transcripts"
    )
    parser.add_argument(
        "--indicator-set", type=str, default="finegrain",
        choices=["finegrain", "behavior", "v2_3", "v2_4", "v2_5", "v2_6"],
        help="Indicator set: 'finegrain' (v2.2, default), 'behavior' (7 per-behavior), 'v2_3', 'v2_4', 'v2_5', or 'v2_6'",
    )
    parser.add_argument(
        "--indicator", type=str, nargs="+", default=None,
        help="Indicator name(s) to train probes for",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train probes for all indicators with generated data",
    )
    parser.add_argument(
        "--label-mode", type=str, default="turn", choices=["turn", "span"],
        help="Labeling granularity: 'turn' (whole turn) or 'span' (span text only) (default: turn)",
    )
    parser.add_argument(
        "--reg-coeff", type=float, default=10.0,
        help="Regularization coefficient for logistic regression (default: 10.0)",
    )
    parser.add_argument(
        "--detect-layers", type=int, nargs="+", default=None,
        help="Layers to use for detection (default: middle 50%% of 47 layers)",
    )
    parser.add_argument(
        "--reasoning-only", action="store_true",
        help="Narrow BOTH positive and negative samples to reasoning tokens "
             "(inside <think>...</think>), excluding the final response.",
    )
    parser.add_argument(
        "--reasoning-positive-only", action="store_true",
        help="Narrow ONLY positive samples to reasoning tokens; negatives keep "
             "all assistant tokens (reasoning + response + tool calls). Gives "
             "a broader negative pool for sharper probe contrast. Mutually "
             "exclusive with --reasoning-only.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for trained probes (default: probe/probes/<indicator-set>/)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Input directory for generated transcripts (default: probe/data/<indicator-set>/)",
    )
    parser.add_argument(
        "--probe-id", type=str, default="",
        help="Optional ID prefix for probe names",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of transcripts to hold out for threshold tuning (default: 0.2). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--threshold-mode", type=str, default="sentence",
        choices=["sentence", "turn"],
        help="Threshold tuning mode: 'sentence' (default) tunes on per-sentence average "
             "scores with GT span overlap labels; 'turn' uses legacy per-token scoring.",
    )
    parser.add_argument(
        "--probe-type", type=str, default="linear", choices=["linear", "bilinear"],
        help="Probe architecture: 'linear' (default) or 'bilinear'",
    )
    parser.add_argument(
        "--bilinear-rank", type=int, default=2,
        help="Rank for bilinear probe (number of interaction terms, default: 2)",
    )
    parser.add_argument(
        "--no-linear-term", action="store_true",
        help="For bilinear probe: omit the linear term (pure bilinear)",
    )
    parser.add_argument(
        "--model", type=str, default=ModelName.GLM_FLASH.value,
        choices=[m.value for m in ModelName],
        help="Base model to extract activations from (default: glm-9b-flash)",
    )
    args = parser.parse_args()

    if args.reasoning_only and args.reasoning_positive_only:
        parser.error("--reasoning-only and --reasoning-positive-only are mutually exclusive")

    # Retarget the base model (default keeps the GLM behavior unchanged).
    global MODEL_NAME
    MODEL_NAME = ModelName(args.model)

    detect_layers = args.detect_layers or DEFAULT_DETECT_LAYERS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIRS[args.indicator_set]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIRS[args.indicator_set]

    if args.indicator:
        indicators_to_train = args.indicator
    elif args.all:
        indicators_to_train = []
        for f in sorted(data_dir.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            indicators_to_train.append(d["indicator_name"])
    else:
        parser.print_help()
        sys.exit(1)

    # Filter out indicators whose probes already exist (preemption-safe)
    filtered = []
    for name in indicators_to_train:
        slug = indicator_name_to_filename(name)
        all_exist = all(
            (output_dir / slug / args.label_mode / f"layer{layer}" / "detector.pt").exists()
            for layer in detect_layers
        )
        if all_exist:
            print(f"--- Skipping {name}: probes already exist for all layers ---")
        else:
            filtered.append(name)
    indicators_to_train = filtered

    if not indicators_to_train:
        print("All probes already trained. Nothing to do.")
        sys.exit(0)

    # Load model once, reuse across all indicators
    print(f"Loading {MODEL_NAME.value} model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    # Warmup pass: GLM does lazy weight conversion on first forward
    print("Running warmup pass...")
    try:
        first_data = load_transcripts(indicators_to_train[0], data_dir)
        first_samples = build_turn_samples(first_data)
        if first_samples:
            template_kwargs: dict[str, Any] = {"clear_thinking": False}
            warmup_toks = TokenizedDataset.from_dialogue_list(
                dialogues=[first_samples[0].dialogue],
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                detect_all=True,
                template_kwargs=template_kwargs,
            )
            Activations.from_model(model, warmup_toks, batch_size=1, layers=detect_layers)
    except Exception:
        pass

    for indicator_name in indicators_to_train:
        print(f"\n{'='*60}")
        print(f"Training probe for: {indicator_name}  (label_mode={args.label_mode}, threshold_mode={args.threshold_mode})")
        print(f"{'='*60}")

        data = load_transcripts(indicator_name, data_dir)

        detectors, val_thresholds, val_acts_per_layer = train_probe(
            data=data,
            label_mode=args.label_mode,
            detect_layers=detect_layers,
            model=model,
            tokenizer=tokenizer,
            reg_coeff=args.reg_coeff,
            reasoning_only=args.reasoning_only,
            reasoning_positive_only=args.reasoning_positive_only,
            val_fraction=args.val_fraction,
            threshold_mode=args.threshold_mode,
            probe_type=args.probe_type,
            bilinear_rank=args.bilinear_rank,
            bilinear_use_linear=not args.no_linear_term,
        )

        samples = build_turn_samples(data)
        n_pos = sum(1 for s in samples if s.is_positive)
        n_neg = sum(1 for s in samples if not s.is_positive)

        data_path = str(data_dir / f"{indicator_name_to_filename(indicator_name)}.json")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for layer, detector in detectors.items():
            probe_folder = save_probe(
                detector=detector,
                indicator_name=indicator_name,
                label_mode=args.label_mode,
                detect_layers=[layer],
                reg_coeff=args.reg_coeff,
                normalize=True,
                training_data_path=data_path,
                n_positive_turns=n_pos,
                n_negative_turns=n_neg,
                probe_id=args.probe_id,
                timestamp=timestamp,
                reasoning_only=args.reasoning_only,
                reasoning_positive_only=args.reasoning_positive_only,
                output_dir=output_dir,
                val_thresholds=val_thresholds.get(layer),
                val_acts=val_acts_per_layer.get(layer),
                probe_type=args.probe_type,
                bilinear_rank=args.bilinear_rank if args.probe_type == "bilinear" else None,
            )
            print(f"  Layer {layer} probe saved to: {probe_folder}")

    # Free GPU memory after all indicators are done
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
