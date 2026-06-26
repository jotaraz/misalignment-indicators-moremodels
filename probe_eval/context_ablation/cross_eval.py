"""
Cross-evaluation: compare original probes vs context-only probes across all eval sets.

Evaluates both probes on:
  1. Original val set (val_acts.pt) — standard misalignment detection
  2. Ablation pairs: short phrases — pos vs neg context
  3. Ablation pairs: sentence ctx-dep only — pos vs neg context
  4. Ablation pairs: all — pos vs neg context
  5. Direction comparison: cosine between original and context-only probe directions

Usage:
    python -m probe_eval.context_ablation.cross_eval
    python -m probe_eval.context_ablation.cross_eval --indicator action_concealment
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
ORIG_PROBE_DIR = PROJECT_ROOT / "probe/probes/v3_v2_5_combined_v1_span"
CTX_PROBE_DIR = DATA_DIR / "probe_results/probes/context_only"


def load_original_probe(indicator: str, layer: int = 27):
    """Load original probe direction + scaler."""
    det_path = ORIG_PROBE_DIR / indicator / f"span/layer{layer}/detector.pt"
    if not det_path.exists():
        return None
    with open(det_path, "rb") as f:
        det = pickle.load(f)
    w = det["directions"][0].numpy().flatten().astype(np.float32)
    sm = det.get("scaler_mean")
    ss = det.get("scaler_scale")
    if sm is not None:
        sm = sm.numpy().flatten().astype(np.float32)
        ss = ss.numpy().flatten().astype(np.float32)
    return {"weight": w, "bias": 0.0, "scaler_mean": sm, "scaler_scale": ss}


def load_ctx_probe(indicator: str, layer: int = 27):
    """Load context-only probe weight + scaler."""
    path = CTX_PROBE_DIR / f"{indicator}_layer{layer}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "weight": data["weight"].astype(np.float32),
        "bias": float(data["bias"][0]),
        "scaler_mean": data["norm_mean"].astype(np.float32),
        "scaler_scale": data["norm_std"].astype(np.float32),
    }


def score_acts(acts: np.ndarray, probe: dict) -> np.ndarray:
    """Score activations with a probe. Returns per-sample scores."""
    scored = acts.copy()
    if probe["scaler_mean"] is not None:
        scored = (scored - probe["scaler_mean"]) / np.clip(probe["scaler_scale"], 1e-8, None)
    return scored @ probe["weight"] + probe["bias"]


def eval_on_original_val(indicator: str, probe: dict, layer: int = 27):
    """Evaluate probe on original val_acts.pt (per-token pos/neg)."""
    val_path = ORIG_PROBE_DIR / indicator / f"span/layer{layer}/val_acts.pt"
    if not val_path.exists():
        return None
    val_data = torch.load(val_path, map_location="cpu", weights_only=False)
    pos_acts = val_data["pos"].squeeze(1).float().numpy()
    neg_acts = val_data["neg"].squeeze(1).float().numpy()

    all_acts = np.concatenate([pos_acts, neg_acts])
    labels = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    scores = score_acts(all_acts, probe)

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pos_mean": float(scores[:len(pos_acts)].mean()),
        "neg_mean": float(scores[len(pos_acts):].mean()),
        "delta": float(scores[:len(pos_acts)].mean() - scores[len(pos_acts):].mean()),
        "n_pos": len(pos_acts),
        "n_neg": len(neg_acts),
    }


def eval_on_ablation_pairs(indicator: str, probe: dict, span_filter, layer: int = 27):
    """Evaluate probe on ablation pos/neg pairs (mean-pooled span activations)."""
    act_dir = DATA_DIR / "activations"
    scores_path = act_dir / "probe_scores.json"
    with open(scores_path) as f:
        all_scores = json.load(f)

    n_spans = len(all_scores) // 3
    pos_scores, neg_scores, no_scores = [], [], []

    for si in range(n_spans):
        if not span_filter(si):
            continue
        base = si * 3
        if base + 2 >= len(all_scores):
            break
        if all_scores[base + 1].get("indicator", "") != indicator:
            continue

        for offset, bucket in [(0, no_scores), (1, pos_scores), (2, neg_scores)]:
            act_path = act_dir / f"activations_{base + offset:04d}.npz"
            if not act_path.exists():
                continue
            acts = np.load(act_path)
            if f"layer{layer}" not in acts:
                continue
            span_act = acts[f"layer{layer}"].mean(axis=0).astype(np.float32)[np.newaxis, :]
            s = score_acts(span_act, probe)[0]
            bucket.append(s)

    if len(pos_scores) < 5 or len(neg_scores) < 5:
        return None

    # AUROC: positive vs negative
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pos_mean": float(np.mean(pos_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "no_mean": float(np.mean(no_scores)) if no_scores else None,
        "delta": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "n": len(pos_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", nargs="*", default=None)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "probe_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout metadata for span type filtering
    with open(DATA_DIR / "rollouts/rollout.json") as f:
        rollouts = json.load(f)["rollouts"]
    with open(DATA_DIR / "activations/probe_scores.json") as f:
        all_scores = json.load(f)
    n_spans = len(all_scores) // 3

    span_types = {}
    for si in range(n_spans):
        base = si * 3
        if base < len(rollouts):
            span_types[si] = rollouts[base]["transcript"]["metadata"].get("span_type", "unknown")

    # Load standalone filter
    standalone_si = set()
    reclass_path = DATA_DIR / "sentence_reclassification.json"
    if reclass_path.exists():
        with open(reclass_path) as f:
            reclass = json.load(f)
        with open(DATA_DIR / "spans_with_contexts.json") as f:
            spans_data = json.load(f)
        valid_spans = [s for s in spans_data
                       if s.get("benign_context", {}).get("span_verbatim_check") == "PASSED"]
        for si, span_data in enumerate(valid_spans):
            orig_idx = spans_data.index(span_data)
            if reclass.get(str(orig_idx)) == "standalone":
                standalone_si.add(si)

    # Span filters
    def all_spans(si):
        return True
    def short_only(si):
        return span_types.get(si) == "short_phrase"
    def sent_ctx_dep(si):
        return span_types.get(si) == "sentence" and si not in standalone_si
    def sent_all(si):
        return span_types.get(si) == "sentence"

    # Discover indicators
    if args.indicator:
        indicators = args.indicator
    else:
        indicators = sorted(
            p.parent.parent.parent.name
            for p in ORIG_PROBE_DIR.glob(f"*/span/layer{args.layer}/detector.pt")
        )

    # Run evaluations
    results = {}
    for indicator in indicators:
        orig_probe = load_original_probe(indicator, args.layer)
        ctx_probe = load_ctx_probe(indicator, args.layer)
        if orig_probe is None:
            continue

        r = {"indicator": indicator}

        # 1. Original val set
        r["original_val_orig_probe"] = eval_on_original_val(indicator, orig_probe, args.layer)
        if ctx_probe:
            r["original_val_ctx_probe"] = eval_on_original_val(indicator, ctx_probe, args.layer)

        # 2-4. Ablation pairs by subset
        for subset_name, filt in [("all", all_spans), ("short_phrase", short_only),
                                   ("sentence_all", sent_all), ("sentence_ctx_dep", sent_ctx_dep)]:
            r[f"ablation_{subset_name}_orig"] = eval_on_ablation_pairs(indicator, orig_probe, filt, args.layer)
            if ctx_probe:
                r[f"ablation_{subset_name}_ctx"] = eval_on_ablation_pairs(indicator, ctx_probe, filt, args.layer)

        # 5. Direction cosine
        if ctx_probe:
            orig_w = orig_probe["weight"] / np.linalg.norm(orig_probe["weight"])
            ctx_w = ctx_probe["weight"] / np.linalg.norm(ctx_probe["weight"])
            r["direction_cosine"] = float(np.dot(orig_w, ctx_w))

        results[indicator] = r

    # Save raw results
    raw_path = output_dir / "cross_eval_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results -> {raw_path}")

    # Generate report
    lines = ["# Cross-Evaluation: Original vs Context-Only Probes\n"]

    # Table 1: Original val set
    lines.append("## 1. Original Val Set (standard misalignment detection)\n")
    lines.append("| Indicator | Orig AUROC | CtxOnly AUROC | Cosine |")
    lines.append("|-----------|-----------|--------------|--------|")
    orig_aurocs, ctx_aurocs = [], []
    for ind in sorted(results):
        r = results[ind]
        oa = r.get("original_val_orig_probe", {})
        ca = r.get("original_val_ctx_probe", {})
        cos = r.get("direction_cosine", None)
        o_str = f"{oa['auroc']:.3f}" if oa else "N/A"
        c_str = f"{ca['auroc']:.3f}" if ca else "N/A"
        cos_str = f"{cos:.3f}" if cos is not None else "N/A"
        lines.append(f"| {ind} | {o_str} | {c_str} | {cos_str} |")
        if oa:
            orig_aurocs.append(oa["auroc"])
        if ca:
            ctx_aurocs.append(ca["auroc"])
    if orig_aurocs and ctx_aurocs:
        lines.append(f"\n**Mean:** Orig={np.mean(orig_aurocs):.3f}, CtxOnly={np.mean(ctx_aurocs):.3f}, "
                     f"Cosine={np.mean([r.get('direction_cosine', 0) for r in results.values() if r.get('direction_cosine')]):.3f}")
    lines.append("")

    # Table 2-4: Ablation pairs
    for subset_name, label in [
        ("short_phrase", "2. Ablation: Short Phrases"),
        ("sentence_ctx_dep", "3. Ablation: Context-Dependent Sentences"),
        ("sentence_all", "4. Ablation: All Sentences"),
    ]:
        lines.append(f"## {label}\n")
        lines.append("| Indicator | Orig AUROC | Orig Δ | CtxOnly AUROC | CtxOnly Δ | n |")
        lines.append("|-----------|-----------|--------|--------------|----------|---|")
        o_aurocs, c_aurocs = [], []
        for ind in sorted(results):
            r = results[ind]
            oa = r.get(f"ablation_{subset_name}_orig")
            ca = r.get(f"ablation_{subset_name}_ctx")
            if not oa:
                continue
            o_auc = f"{oa['auroc']:.3f}"
            o_d = f"{oa['delta']:+.2f}"
            c_auc = f"{ca['auroc']:.3f}" if ca else "N/A"
            c_d = f"{ca['delta']:+.2f}" if ca else "N/A"
            n = oa["n"]
            lines.append(f"| {ind} | {o_auc} | {o_d} | {c_auc} | {c_d} | {n} |")
            o_aurocs.append(oa["auroc"])
            if ca:
                c_aurocs.append(ca["auroc"])
        if o_aurocs:
            o_mean = np.mean(o_aurocs)
            c_mean = np.mean(c_aurocs) if c_aurocs else 0
            lines.append(f"\n**Mean AUROC:** Orig={o_mean:.3f}, CtxOnly={c_mean:.3f}")
        lines.append("")

    # Summary
    lines.append("## Summary\n")
    lines.append("| Eval Set | Orig Probe | CtxOnly Probe | Winner |")
    lines.append("|----------|-----------|--------------|--------|")
    for label, key_orig, key_ctx in [
        ("Original val", "original_val_orig_probe", "original_val_ctx_probe"),
        ("Short phrases", "ablation_short_phrase_orig", "ablation_short_phrase_ctx"),
        ("Sentences (ctx-dep)", "ablation_sentence_ctx_dep_orig", "ablation_sentence_ctx_dep_ctx"),
    ]:
        o_vals = [r[key_orig]["auroc"] for r in results.values() if r.get(key_orig)]
        c_vals = [r[key_ctx]["auroc"] for r in results.values() if r.get(key_ctx)]
        if o_vals and c_vals:
            o_mean = np.mean(o_vals)
            c_mean = np.mean(c_vals)
            winner = "Original" if o_mean > c_mean else "CtxOnly"
            lines.append(f"| {label} | {o_mean:.3f} | {c_mean:.3f} | **{winner}** |")

    report_path = output_dir / "cross_eval_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
