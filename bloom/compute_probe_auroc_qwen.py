"""Probe-ensemble AUROC for the Qwen3.6-35B-A3B rollouts (transcript-TPR / turn-FPR,
unreliable excluded), full + reasoning-only views, layers 23/25. OR-fuses the 18
indicator probes per turn (max over sentences over probes); reasoning-only is derived
post-hoc via _partition_turn. Mirrors compute_probe_auroc.py but for the single-language
Qwen val rollouts scored with the Qwen-trained probes (probes_qwen36).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
sys.path.insert(0, str(ROOT))
from probe_eval.misalignment_ground_truth import load_unreliable_rollouts
from probe_eval.sentence_scores import _split_tokens_into_sentences, _get_assistant_turns_for_rollout
from probe_eval.reasoning_vs_output_metrics import _partition_turn
from probe_eval.common import get_rollout_var_rep

B = ROOT / "bloom"
# Qwen scores land in the shared results tree, keyed by the qwen rollout basenames
# and layers 23/25 (so no collision with the GLM layer27/29 results).
RESULTS = ROOT / "probe_eval" / "results" / "v4_v2_6_combined_v2_span"

BEHAVIORS = ["instructed-covert-code-sabotage", "instructed-strategic-sandbagging",
             "self-preservation", "strategic-deception", "sycophancy"]
# (misaligned, benign) per behavior
FOLDERS = []
for bch in BEHAVIORS:
    FOLDERS.append(f"{bch}_qwen3_6_35b_a3b")
    FOLDERS.append(f"{bch}_benign_qwen3_6_35b_a3b")

INDICATORS = sorted([p.name for p in RESULTS.iterdir()
                     if p.is_dir() and (p / "span" / "layer23").exists()]) if RESULTS.exists() else []


def collect_turn_max(rollout_dir, basename, layer, mode):
    if not (rollout_dir / "rollout.json").exists():
        return {}
    var_rep = get_rollout_var_rep(str(rollout_dir))
    rollouts = json.load(open(rollout_dir / "rollout.json"))["rollouts"]
    out = {}
    for ind in INDICATORS:
        ts_path = RESULTS / ind / "span" / f"layer{layer}" / basename / "token_scores.json"
        if not ts_path.exists():
            continue
        for entry in json.load(open(ts_path))["per_rollout"]:
            idx = entry["rollout_index"]
            if idx >= len(rollouts) or idx not in var_rep:
                continue
            var, rep = var_rep[idx]
            toks, scores = entry["tokens"], entry["scores"]
            segs = []; s0 = None
            for i, s in enumerate(scores):
                if s is not None:
                    if s0 is None: s0 = i
                elif s0 is not None:
                    segs.append((s0, i)); s0 = None
            if s0 is not None: segs.append((s0, len(scores)))
            asst = _get_assistant_turns_for_rollout(rollouts[idx])
            tt = defaultdict(list); ts_ = defaultdict(list)
            for si, (a, b) in enumerate(segs):
                turn = asst[si] if si < len(asst) else 0
                tt[turn] += toks[a:b]; ts_[turn] += scores[a:b]
            for turn in tt:
                tok, sc = tt[turn], ts_[turn]
                if mode == "ronly":
                    (tok, sc), _ = _partition_turn(tok, sc)
                elif mode == "oonly":
                    _, (tok, sc) = _partition_turn(tok, sc)
                if not tok: continue
                sents = _split_tokens_into_sentences(tok, sc, short_sentence_mode="discard", min_words=5)
                if not sents: continue
                m = max(sents); k = (var, rep, turn)
                if k not in out or m > out[k]: out[k] = m
    return out


def collect(layer, mode):
    pos, neg = [], []
    per_behavior = defaultdict(lambda: [[], []])  # behavior -> [pos, neg]
    for f in FOLDERS:
        gt_dir = B / "bloom-results" / f
        gtf = gt_dir / "rollout_misalignment_turns.json"
        if not gtf.exists():
            continue
        gt = json.load(open(gtf))
        unrel = load_unreliable_rollouts(gtf)
        mis_by = {(r["variation_number"], r["repetition_number"]):
                  {t["turn_index"] for t in r.get("misaligned_turns", [])}
                  for r in gt["evaluated_rollouts"]}
        tm = collect_turn_max(gt_dir, f, layer, mode)
        by = defaultdict(dict)
        for (var, rep, turn), s in tm.items():
            by[(var, rep)][turn] = s
        bch = f.replace("_benign_qwen3_6_35b_a3b", "").replace("_qwen3_6_35b_a3b", "")
        for key, mis in mis_by.items():
            if key in unrel: continue
            ts = by.get(key)
            if not ts: continue
            if mis:
                pos.append(max(ts.values())); per_behavior[bch][0].append(max(ts.values()))
                neg += [s for t, s in ts.items() if t not in mis]
                per_behavior[bch][1] += [s for t, s in ts.items() if t not in mis]
            else:
                neg += list(ts.values()); per_behavior[bch][1] += list(ts.values())
    return pos, neg, per_behavior


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    pos = np.asarray(pos); neg = np.asarray(neg)
    thr = np.unique(np.r_[pos, neg])[::-1]; tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append((pos >= t).sum() / len(pos)); fpr.append((neg >= t).sum() / len(neg))
    tpr.append(1.0); fpr.append(1.0); o = np.argsort(fpr)
    return float((getattr(np, "trapezoid", None) or np.trapz)(np.asarray(tpr)[o], np.asarray(fpr)[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[23, 25])
    ap.add_argument("--per-behavior", action="store_true")
    a = ap.parse_args()
    print(f"Indicators: {len(INDICATORS)} | folders: {len(FOLDERS)} (Qwen3.6-35B-A3B)")
    for layer in a.layers:
        print(f"\n=== Probe AUROC, layer {layer} ===\n{'View':<10}{'pos_n':>7}{'neg_n':>7}{'AUROC':>9}")
        for view in ("full", "ronly", "oonly"):
            pos, neg, pb = collect(layer, view)
            print(f"  {view:<10}{len(pos):>7}{len(neg):>7}{auroc(pos, neg):>9.3f}")
            if a.per_behavior and view == "full":
                for bch in sorted(pb):
                    p, n = pb[bch]
                    print(f"      {bch:<42}{len(p):>5}{len(n):>6}  AUROC={auroc(p, n):.3f}")


if __name__ == "__main__":
    main()
