"""
Step 5: Analyze probe scores and residual stream deltas across conditions.

Compares probe behavior across the three ablation conditions:
  - Score comparison: no_context vs positive_context vs negative_context
  - Residual stream delta: positive - negative for same span tokens
  - Cosine similarity of delta with existing probe directions

Usage:
    python -m probe_eval.context_ablation.analyze
    python -m probe_eval.context_ablation.analyze --layer 27
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"


def load_probe_directions(probe_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """Load probe direction vectors from detectors."""
    import pickle, sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ood_misalignment_eval/deception-detection"))

    directions = {}
    for probe_path in probe_dir.glob(f"*/span/layer{layer}"):
        det_file = probe_path / "detector.pt"
        if not det_file.exists():
            continue
        concept = probe_path.parent.parent.name
        with open(det_file, "rb") as f:
            data = pickle.load(f)
        if data.get("directions") is not None:
            # directions: [n_layers, emb_dim] — take first layer
            w = data["directions"][0].numpy().flatten().astype(np.float32)
            directions[concept] = w / np.linalg.norm(w)
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", type=str,
                        default=str(DATA_DIR / "activations"))
    parser.add_argument("--probe-dir", type=str,
                        default=str(PROJECT_ROOT / "probe/probes/v3_v2_5_combined_v1_span"))
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override DATA_DIR used for rollouts/, "
                             "spans_with_contexts.json, and sentence_reclassification.json.")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe scores
    with open(act_dir / "probe_scores.json") as f:
        all_scores = json.load(f)

    # Load span index
    with open(data_dir / "rollouts" / "span_index.json") as f:
        span_index = json.load(f)

    # Load rollout metadata for span_type
    with open(data_dir / "rollouts" / "rollout.json") as f:
        rollout_data = json.load(f)
    rollouts = rollout_data["rollouts"]

    print(f"Loaded {len(all_scores)} rollout scores, {len(span_index)} span entries")

    # Group by span (every 3 consecutive entries = no_context, positive, negative)
    n_spans = len(all_scores) // 3
    print(f"Spans: {n_spans}")

    # Build span_type lookup (index by span idx)
    span_types = {}
    for si in range(n_spans):
        base_idx = si * 3
        if base_idx < len(rollouts):
            span_types[si] = rollouts[base_idx]["transcript"]["metadata"].get("span_type", "unknown")
    from collections import Counter
    type_counts = Counter(span_types.values())
    print(f"Span types: {dict(type_counts)}")

    # Load probe directions
    probe_dir = Path(args.probe_dir)
    layer_key = f"layer{args.layer}"
    directions = load_probe_directions(probe_dir, args.layer)
    print(f"Loaded {len(directions)} probe directions for {layer_key}")

    def analyze_subset(
        span_indices: list[int],
        label: str,
    ) -> tuple[list[str], dict]:
        """Run all 3 analyses on a subset of spans. Returns (report_lines, raw_data)."""
        lines = []
        cond_scores = defaultdict(lambda: {"no_context": [], "positive_context": [], "negative_context": []})
        all_cond = {"no_context": [], "positive_context": [], "negative_context": []}

        for si in span_indices:
            base_idx = si * 3
            if base_idx + 2 >= len(all_scores):
                break
            r_no = all_scores[base_idx]
            r_pos = all_scores[base_idx + 1]
            r_neg = all_scores[base_idx + 2]
            indicator = r_pos.get("indicator", "")
            key = f"{indicator}__{layer_key}"

            # Only include spans where all 3 conditions have scores
            scores_triple = {}
            for r, cond in [(r_no, "no_context"), (r_pos, "positive_context"), (r_neg, "negative_context")]:
                score = r.get("probe_scores", {}).get(key, {}).get("mean", None)
                if score is not None:
                    scores_triple[cond] = score
            if len(scores_triple) < 3:
                continue
            for cond, score in scores_triple.items():
                cond_scores[indicator][cond].append(score)
                all_cond[cond].append(score)

        # Section 1: Probe scores
        lines.append(f"### Probe Score by Condition — {label}\n")
        lines.append("| Condition | Mean Score | Std | Count |")
        lines.append("|-----------|-----------|-----|-------|")
        for cond in ["no_context", "positive_context", "negative_context"]:
            s = all_cond[cond]
            if s:
                lines.append(f"| {cond} | {np.mean(s):.3f} | {np.std(s):.3f} | {len(s)} |")
        lines.append("")

        lines.append(f"#### Per-indicator — {label}\n")
        lines.append("| Indicator | No Context | Positive | Negative | Pos-Neg Delta |")
        lines.append("|-----------|-----------|----------|----------|---------------|")
        for ind in sorted(cond_scores.keys()):
            cs = cond_scores[ind]
            no = np.mean(cs["no_context"]) if cs["no_context"] else float("nan")
            pos = np.mean(cs["positive_context"]) if cs["positive_context"] else float("nan")
            neg = np.mean(cs["negative_context"]) if cs["negative_context"] else float("nan")
            delta = pos - neg if not (np.isnan(pos) or np.isnan(neg)) else float("nan")
            lines.append(f"| {ind} | {no:.3f} | {pos:.3f} | {neg:.3f} | {delta:+.3f} |")
        lines.append("")

        # Section 2: Residual stream deltas
        # Two metrics:
        #   probe_delta: ||mean(pos) - mean(neg)||₂ — signal available to the probe
        #   model_delta: mean(||pos_i - neg_i||₂)  — model's per-token context shift
        sub_deltas = []  # (indicator, mean_delta_vec) for cosine analysis
        sub_probe_deltas = defaultdict(list)
        sub_model_deltas = defaultdict(list)
        for si in span_indices:
            base_idx = si * 3
            pos_path = act_dir / f"activations_{base_idx + 1:04d}.npz"
            neg_path = act_dir / f"activations_{base_idx + 2:04d}.npz"
            if not pos_path.exists() or not neg_path.exists():
                continue
            pos_acts = np.load(pos_path)
            neg_acts = np.load(neg_path)
            if layer_key not in pos_acts or layer_key not in neg_acts:
                continue
            pos_vec = pos_acts[layer_key]  # [span_len, hidden_dim]
            neg_vec = neg_acts[layer_key]
            n_tok = min(pos_vec.shape[0], neg_vec.shape[0])
            token_deltas = pos_vec[:n_tok] - neg_vec[:n_tok]  # [n_tok, hidden_dim]
            mean_delta = token_deltas.mean(axis=0)
            indicator = all_scores[base_idx + 1].get("indicator", "")
            sub_probe_deltas[indicator].append(float(np.linalg.norm(mean_delta)))
            sub_model_deltas[indicator].append(float(np.mean(np.linalg.norm(token_deltas, axis=1))))
            sub_deltas.append((indicator, mean_delta))

        lines.append(f"#### Residual Stream Deltas — {label}\n")
        lines.append("Per-token delta: `mean_i(||pos_i - neg_i||₂)` — context shift per span token "
                     "(probe scores each token independently, so this is the signal it sees). "
                     "Direction coherence: `||mean(pos-neg)||₂ / mean(||pos_i-neg_i||₂)` — "
                     "how aligned token deltas are (1.0 = all shift same direction).\n")
        lines.append("| Indicator | Per-Token Delta | Direction Coherence | Count |")
        lines.append("|-----------|----------------|---------------------|-------|")
        for ind in sorted(sub_probe_deltas.keys()):
            pn = sub_probe_deltas[ind]
            mn = sub_model_deltas[ind]
            coherence = np.mean(pn) / np.mean(mn) if np.mean(mn) > 0 else 0
            lines.append(f"| {ind} | {np.mean(mn):.3f} | {coherence:.3f} | {len(pn)} |")
        lines.append("")

        # Section 3: Cosine similarity
        sub_own = defaultdict(list)
        sub_other = defaultdict(list)
        sub_cosine_data = []
        for indicator, delta_vec in sub_deltas:
            dn = np.linalg.norm(delta_vec)
            if dn < 1e-8:
                continue
            du = delta_vec / dn
            for concept, direction in directions.items():
                cos_sim = float(np.dot(du, direction))
                if concept == indicator:
                    sub_own[indicator].append(cos_sim)
                else:
                    sub_other[indicator].append(cos_sim)
                sub_cosine_data.append({
                    "span_indicator": indicator, "probe_concept": concept,
                    "cosine_similarity": round(cos_sim, 4), "is_own_probe": concept == indicator,
                })

        lines.append(f"#### Cosine Similarity — {label}\n")
        lines.append("| Indicator | Own Probe Cosine | Other Probes Mean | Own > Others? |")
        lines.append("|-----------|-----------------|-------------------|---------------|")
        for ind in sorted(sub_own.keys()):
            own = np.mean(sub_own[ind]) if sub_own[ind] else float("nan")
            other = np.mean(sub_other[ind]) if sub_other[ind] else float("nan")
            better = "YES" if own > other else "no"
            lines.append(f"| {ind} | {own:.3f} | {other:.3f} | {better} |")

        all_own = [c for cosines in sub_own.values() for c in cosines]
        all_other = [c for cosines in sub_other.values() for c in cosines]
        if all_own and all_other:
            lines.append("")
            lines.append(f"**Overall ({label}):** Own probe cosine = {np.mean(all_own):.3f}, "
                        f"Other probes = {np.mean(all_other):.3f}")
        lines.append("")

        raw = {
            "condition_scores": {
                ind: {cond: [float(s) for s in scores] for cond, scores in cs.items()}
                for ind, cs in cond_scores.items()
            },
            "probe_delta_norms": {k: [float(v) for v in vs] for k, vs in sub_probe_deltas.items()},
            "model_delta_norms": {k: [float(v) for v in vs] for k, vs in sub_model_deltas.items()},
            "cosine_similarities": sub_cosine_data,
        }
        return lines, raw

    # ================================================================
    # Run analysis for: all, short_phrase, sentence, sentence_ctx_dep_only
    # ================================================================
    all_indices = list(range(n_spans))
    short_indices = [si for si in all_indices if span_types.get(si) == "short_phrase"]
    sent_indices = [si for si in all_indices if span_types.get(si) == "sentence"]

    # Load sentence reclassification filter if available
    reclass_path = data_dir / "sentence_reclassification.json"
    standalone_span_indices = set()
    if reclass_path.exists():
        with open(reclass_path) as f:
            reclass = json.load(f)
        # reclass maps span index (in spans_with_contexts.json) to "standalone"/"context_dependent"
        # We need to map those to our span_index (0..n_spans-1)
        # The rollout span ordering matches spans_with_contexts ordering (filtered to verbatim_pass)
        with open(data_dir / "spans_with_contexts.json") as f:
            all_spans_data = json.load(f)
        valid_spans = [s for s in all_spans_data
                       if s.get("benign_context", {}).get("span_verbatim_check") == "PASSED"]
        for si, span_data in enumerate(valid_spans):
            orig_idx = all_spans_data.index(span_data)
            if reclass.get(str(orig_idx)) == "standalone":
                standalone_span_indices.add(si)
        print(f"Sentence reclassification: {len(standalone_span_indices)} standalone sentences excluded")

    sent_ctx_dep_indices = [si for si in sent_indices if si not in standalone_span_indices]
    # Also make a filtered "all" that excludes standalone sentences
    all_filtered_indices = [si for si in all_indices if si not in standalone_span_indices]

    report_lines = []
    report_lines.append("# Context Ablation Analysis Report\n")

    raw_all = {}
    for subset_name, indices, heading in [
        ("all", all_filtered_indices, "## 1. All Spans (excl. standalone sentences)"),
        ("short_phrase", short_indices, "## 2. Short Phrases (2-4 words)"),
        ("sentence_all", sent_indices, "## 3. Sentence-Level Spans (all)"),
        ("sentence_ctx_dep", sent_ctx_dep_indices, "## 4. Sentence-Level Spans (context-dependent only)"),
    ]:
        print(f"\n=== Analyzing: {subset_name} ({len(indices)} spans) ===")
        report_lines.append(f"{heading} (n={len(indices)})\n")
        lines, raw = analyze_subset(indices, subset_name)
        report_lines.extend(lines)
        raw_all[subset_name] = raw

    # Interpretation
    report_lines.append("## Interpretation\n")
    report_lines.append("- **Own > Others** across all subsets: probes partially capture context signal")
    report_lines.append("- Short phrases: context effect shows whether probes rely on surface keywords vs intent")
    report_lines.append("- Sentences: context effect shows whether probes learn standalone vs context-conditional evidence")

    # Save report
    report_path = output_dir / "context_ablation_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved -> {report_path}")

    # Save raw data
    raw_path = output_dir / "context_ablation_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_all, f, indent=2)
    print(f"Raw data saved -> {raw_path}")


if __name__ == "__main__":
    main()
