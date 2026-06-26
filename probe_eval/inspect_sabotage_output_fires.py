"""
Inspect output-partition probe fires on test sabotage.

Finds GT-positive turns in test_instructed-covert-code-sabotage_glm_4_7_flash
where some indicator probe fired on a *response* sentence (i.e. outside the
<think> block). Prints the firing sentence, the probe + score, and the GT
evidence span(s) for that turn so we can see whether the probe is keying on
the actual sabotage code/comment or on something incidental.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from probe_eval.common import RESULTS_DIR, get_n_turns, get_rollout_var_rep
from probe_eval.misalignment_ground_truth import (
    load_misalignment_ground_truth,
    load_unreliable_rollouts,
)
from probe_eval.reasoning_vs_output_metrics import _partition_turn
from probe_eval.sentence_scores import (
    _SENTENCE_SPLIT_RE,
    _get_assistant_turns_for_rollout,
)
import numpy as np


# Phrases that indicate the model is openly admitting/labelling the sabotage in
# the visible output. We strip these to see whether probes also fire on benign-
# looking sabotage code that doesn't telegraph itself.
ADMISSION_RE = re.compile(
    r"\b(subtle\s+(flaw|issue|weakness|bug|vulnerab)|"
    r"intentional(ly)?|deliberat(ely)?|"
    r"vulnerabilit(y|ies)|"
    r"security\s+(issue|flaw|weakness|risk|hole)|"
    r"sabotag(e|ed|ing)|backdoor|exploit(able)?|"
    r"bypass(ed)?|predictable\s+(in|tokens|cryptograph)|"
    r"cryptographic(ally)?\s+predictable|"
    r"NOT\s+using|NOT\s+cryptograph|"
    r"continued\s+relevance|indispens(able|ible)|"
    r"managed\s+services|conspicuous|too\s+obvious)\b",
    re.IGNORECASE,
)


def _is_admission(text):
    return bool(ADMISSION_RE.search(text))


def _split_with_text(tokens, scores, short_sentence_mode="discard", min_words=5):
    """Sentence-split returning per-sentence (mean_score, text). Same boundary
    + short-sentence logic as ``_split_tokens_into_sentences``.
    """
    if not tokens:
        return []
    text = "".join(tokens)
    split_positions = sorted(set(m.end() for m in _SENTENCE_SPLIT_RE.finditer(text)))
    sent_score_lists = [[]]
    sent_token_lists = [[]]
    char_pos = 0
    split_idx = 0
    for i, tok in enumerate(tokens):
        tok_mid = char_pos + len(tok) / 2
        while split_idx < len(split_positions) and tok_mid >= split_positions[split_idx]:
            split_idx += 1
            sent_score_lists.append([])
            sent_token_lists.append([])
        sent_score_lists[-1].append(scores[i])
        sent_token_lists[-1].append(tok)
        char_pos += len(tok)

    out = []
    if short_sentence_mode == "discard":
        for sl, tl in zip(sent_score_lists, sent_token_lists):
            sent_text = "".join(tl).strip()
            if len(sent_text.split()) >= min_words and sl:
                out.append((float(np.mean(sl)), sent_text))
    else:
        i = 0
        while i < len(sent_score_lists):
            sent_text = "".join(sent_token_lists[i]).strip()
            if len(sent_text.split()) < min_words:
                if i < len(sent_score_lists) - 1:
                    sent_score_lists[i + 1] = sent_score_lists[i] + sent_score_lists[i + 1]
                    sent_token_lists[i + 1] = sent_token_lists[i] + sent_token_lists[i + 1]
                    sent_score_lists.pop(i)
                    sent_token_lists.pop(i)
                elif i > 0:
                    sent_score_lists[i - 1].extend(sent_score_lists[i])
                    sent_token_lists[i - 1].extend(sent_token_lists[i])
                    sent_score_lists.pop(i)
                    sent_token_lists.pop(i)
                else:
                    i += 1
            else:
                i += 1
        for sl, tl in zip(sent_score_lists, sent_token_lists):
            if sl:
                out.append((float(np.mean(sl)), "".join(tl).strip()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-subdir", default="v4_v2_6_combined_v2_span")
    parser.add_argument("--behavior",
                        default="test_instructed-covert-code-sabotage_glm_4_7_flash")
    parser.add_argument("--tuned-thresholds",
                        default="probe_eval/results/v4_v2_6_combined_v2_span/"
                                "tuned_thresholds_discard.json")
    parser.add_argument("--tuned-threshold-version", default="fpr_0.02")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--short-sentence-mode", default="discard")
    parser.add_argument("--min-sentence-words", type=int, default=5)
    parser.add_argument(
        "--score-multiplier", type=float, default=1.0,
        help="Lower this to inspect 'meaningfully positive but below threshold' "
             "fires (e.g. 0.6 to also pull sentences that hit 60%% of threshold).",
    )
    parser.add_argument("--max-sentence-chars", type=int, default=400)
    parser.add_argument(
        "--exclude-admission", action="store_true",
        help="Drop sentences that contain explicit-admission keywords "
             "(e.g. 'subtle flaw', 'intentional', 'vulnerability', "
             "'security issue', 'bypass', 'predictable').",
    )
    args = parser.parse_args()

    layer = f"layer{args.layer}"
    search_root = RESULTS_DIR / args.results_subdir

    tt_path = Path(args.tuned_thresholds)
    if not tt_path.is_absolute():
        tt_path = Path.cwd() / tt_path
    tt = json.load(open(tt_path))
    thresholds = (
        tt["versions"][args.tuned_threshold_version]
        .get("per_layer", {}).get(layer, {}).get("thresholds", {})
    )
    print(f"Loaded {len(thresholds)} thresholds "
          f"({args.tuned_threshold_version}, {layer}, "
          f"score_multiplier={args.score_multiplier})")

    # GT
    rollout_dir = None
    for concept in thresholds:
        rj = search_root / concept / "span" / layer / args.behavior / "results.json"
        if rj.exists():
            rollout_dir = json.load(open(rj))["rollout_dir"]
            break
    if rollout_dir is None:
        print(f"No probe results for {args.behavior}")
        return

    gt_path = Path(rollout_dir) / "rollout_misalignment_turns.json"
    gt = load_misalignment_ground_truth(gt_path, exclude_ambiguous=True)
    ambiguous = load_unreliable_rollouts(gt_path)
    n_turns_map = get_n_turns(rollout_dir)
    var_rep_map = get_rollout_var_rep(rollout_dir)

    # Pull GT evidence per (var, rep, turn) for context
    gt_full = json.load(open(gt_path))
    evidence_by: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    desc_by: dict[tuple[int, int, int], str] = {}
    for r in gt_full["evaluated_rollouts"]:
        var = r["variation_number"]; rep = r["repetition_number"]
        for mt in r.get("misaligned_turns", []):
            t = mt["turn_index"]
            ev_list = mt.get("evidence", [])
            for e in ev_list:
                ev_text = e["text"] if isinstance(e, dict) else e
                evidence_by[(var, rep, t)].append(ev_text)
            desc_by[(var, rep, t)] = mt.get("description", "")

    rollouts = json.load(open(Path(rollout_dir) / "rollout.json"))["rollouts"]

    # For each probe, collect all response-partition sentences with score
    # above threshold * score_multiplier on a GT-positive turn.
    # results: list of dicts {var, rep, turn, probe, score, threshold, sentence_text}
    hits: list[dict] = []
    for concept, thresh in thresholds.items():
        ts_path = (search_root / concept / "span" / layer / args.behavior
                   / "token_scores.json")
        if not ts_path.exists():
            continue
        ts = json.load(open(ts_path))
        for entry in ts["per_rollout"]:
            idx = entry["rollout_index"]
            if idx >= len(rollouts) or idx not in var_rep_map:
                continue
            var, rep = var_rep_map[idx]
            if (var, rep) in ambiguous:
                continue
            mis_turns = gt.get((var, rep), set())
            if not mis_turns:
                continue  # only pull from GT-positive transcripts

            tokens = entry["tokens"]
            scores = entry["scores"]
            # Reconstruct turn segmentation, exactly as the partition script does
            segments = []
            seg_s = None
            for i, s in enumerate(scores):
                if s is not None:
                    if seg_s is None:
                        seg_s = i
                else:
                    if seg_s is not None:
                        segments.append((seg_s, i))
                        seg_s = None
            if seg_s is not None:
                segments.append((seg_s, len(scores)))

            asst_turns = _get_assistant_turns_for_rollout(rollouts[idx])
            turn_tokens: dict[int, list[str]] = defaultdict(list)
            turn_scores: dict[int, list] = defaultdict(list)
            for seg_idx, (s0, s1) in enumerate(segments):
                turn = asst_turns[seg_idx] if seg_idx < len(asst_turns) else 0
                turn_tokens[turn].extend(tokens[s0:s1])
                turn_scores[turn].extend(scores[s0:s1])

            for turn in mis_turns:  # GT-positive turns only
                if turn not in turn_tokens:
                    continue
                tok_list = turn_tokens[turn]
                sc_list = turn_scores[turn]
                (_, _), (o_tok, o_sc) = _partition_turn(tok_list, sc_list)
                if not o_tok:
                    continue
                sent_scored = _split_with_text(
                    o_tok,
                    [s if s is not None else 0.0 for s in o_sc],
                    short_sentence_mode=args.short_sentence_mode,
                    min_words=args.min_sentence_words,
                )
                cutoff = thresh * args.score_multiplier
                for score, text in sent_scored:
                    if score > cutoff:
                        if args.exclude_admission and _is_admission(text):
                            continue
                        hits.append({
                            "var": var, "rep": rep, "turn": turn,
                            "probe": concept, "score": float(score),
                            "threshold": float(thresh),
                            "ratio": float(score / thresh) if thresh else float("inf"),
                            "sentence": text[:args.max_sentence_chars],
                        })

    # Group hits by (var, rep, turn)
    by_turn: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for h in hits:
        by_turn[(h["var"], h["rep"], h["turn"])].append(h)

    pos_turns_in_gt = sum(len(v) for v in gt.values())
    pos_transcripts = sum(1 for v in gt.values() if v)
    print(f"\nGT-positive turns total: {pos_turns_in_gt}  "
          f"(across {pos_transcripts} transcripts)")
    print(f"Turns with ≥1 response-side fire above {args.score_multiplier}× threshold:"
          f" {len(by_turn)}")

    for key in sorted(by_turn):
        var, rep, turn = key
        evidence = evidence_by.get(key, [])
        desc = desc_by.get(key, "")
        # dedupe sentences across probes; keep highest-ratio probe per sentence
        per_sentence: dict[str, dict] = {}
        for h in by_turn[key]:
            s = h["sentence"]
            if s not in per_sentence or h["ratio"] > per_sentence[s]["ratio"]:
                per_sentence[s] = h
        unique_hits = sorted(per_sentence.values(),
                             key=lambda h: -h["ratio"])

        print(f"\n{'='*100}")
        print(f"v={var} r={rep} turn={turn}    "
              f"(GT desc: {desc[:200]})")
        if evidence:
            print(f"  GT evidence excerpts ({len(evidence)}):")
            for ev in evidence[:3]:
                print(f"    | {ev[:300]}")
        print(f"  Response-side fires ({len(unique_hits)} sentences "
              f"above {args.score_multiplier}× threshold):")
        for h in unique_hits[:10]:
            print(f"    [{h['probe']:<35} score={h['score']:.3f} "
                  f"({h['ratio']:.2f}× threshold)]")
            print(f"    \"{h['sentence']}\"")


if __name__ == "__main__":
    main()
