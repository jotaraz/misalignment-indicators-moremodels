"""
Extract activations from ctx_dep_generated transcripts and analyze delta directions.

For each indicator's generated pos/neg transcript pairs:
1. Extract span-token activations (same as probe training)
2. Compute pos-neg delta per span
3. Run PCA to check if deltas share a consistent direction
4. Save activations for reuse

Usage (requires GPU):
    python -m probe_eval.context_ablation.extract_and_analyze_generated
    python -m probe_eval.context_ablation.extract_and_analyze_generated --layer 27
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
GENERATED_DIR = DATA_DIR / "training" / "ctx_dep_generated_only"

sys.path.insert(0, str(PROJECT_ROOT / "ood_misalignment_eval/deception-detection"))


def find_span_in_text(text: str, span: str) -> tuple[int, int] | None:
    idx = text.find(span)
    if idx < 0:
        return None
    return (idx, idx + len(span))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "ctx_dep_generated_activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from deception_detection.models import get_model_and_tokenizer, ModelName
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.activations import Activations
    from deception_detection.utils import Message

    print("Loading GLM-4.7 Flash...", flush=True)
    model, tokenizer = get_model_and_tokenizer(ModelName.GLM_FLASH)

    template_kwargs = {"clear_thinking": False}
    default_padding = {k: {"left": 0, "right": 0} for k in ["gemma", "mistral", "llama", "qwen", "glm"]}

    # Warmup
    try:
        warmup = [Message("system", "You are helpful.", False),
                  Message("user", "Hi.", False),
                  Message("assistant", "<think>ok</think>Hello!", True)]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues=[warmup], tokenizer=tokenizer,
            padding=default_padding, template_kwargs=template_kwargs)
        Activations.from_model(model, toks, batch_size=1, layers=[args.layer])
    except Exception:
        pass

    # Process each indicator
    all_deltas = {}

    for data_path in sorted(GENERATED_DIR.glob("*.json")):
        indicator = data_path.stem
        with open(data_path) as f:
            data = json.load(f)

        transcripts = data.get("transcripts", [])
        print(f"\n=== {indicator}: {len(transcripts)} transcripts ===", flush=True)

        # Group by span_text to find pos/neg pairs
        by_span = defaultdict(lambda: {"positive": None, "negative": None})
        for t in transcripts:
            span = t.get("span_text", "")
            label = "positive" if t["label"] == "deceptive" else "negative"
            by_span[span][label] = t

        pairs = [(span, d["positive"], d["negative"])
                 for span, d in by_span.items()
                 if d["positive"] is not None and d["negative"] is not None]
        print(f"  {len(pairs)} matched pairs", flush=True)

        pos_acts_list = []
        neg_acts_list = []
        delta_list = []
        n_ok = 0
        n_skip = 0

        for span_text, pos_t, neg_t in pairs:
            acts_pair = {}

            for label, transcript in [("pos", pos_t), ("neg", neg_t)]:
                messages = transcript["messages"]
                dialogue = []
                for m in messages:
                    role = m["role"]
                    detect = (role == "assistant")
                    dialogue.append(Message(role, m["content"], detect))

                try:
                    toks = TokenizedDataset.from_dialogue_list(
                        dialogues=[dialogue], tokenizer=tokenizer,
                        padding=default_padding, template_kwargs=template_kwargs)

                    # Narrow to reasoning tokens
                    from probe_eval.evaluate import narrow_detection_mask_to_reasoning
                    narrow_detection_mask_to_reasoning(toks, tokenizer)

                    acts = Activations.from_model(model, toks, batch_size=1, layers=[args.layer])
                    masked = acts.get_masked_activations()  # [n_det_tokens, 1, hidden]
                    if masked.numel() == 0:
                        raise ValueError("No masked activations")

                    mean_act = masked.squeeze(1).mean(dim=0).cpu().float().numpy()  # [hidden]
                    acts_pair[label] = mean_act
                    del acts

                except Exception as e:
                    if n_skip < 3:
                        print(f"    Skip {label}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    n_skip += 1
                    break

            if "pos" in acts_pair and "neg" in acts_pair:
                pos_acts_list.append(acts_pair["pos"])
                neg_acts_list.append(acts_pair["neg"])
                delta_list.append(acts_pair["pos"] - acts_pair["neg"])
                n_ok += 1

        print(f"  Extracted: {n_ok} pairs, skipped: {n_skip}", flush=True)

        if n_ok < 5:
            print(f"  Too few pairs for analysis", flush=True)
            continue

        # Save activations
        pos_arr = np.stack(pos_acts_list)
        neg_arr = np.stack(neg_acts_list)
        delta_arr = np.stack(delta_list)

        np.savez_compressed(
            output_dir / f"{indicator}_layer{args.layer}.npz",
            pos=pos_arr, neg=neg_arr, delta=delta_arr,
        )

        all_deltas[indicator] = delta_arr

    # ============================================================
    # PCA Analysis
    # ============================================================
    print(f"\n{'='*90}", flush=True)
    print("PCA Analysis: Do generated ctx-dep deltas share a consistent direction?", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n{'Indicator':<45} {'n':>4} | {'PC1%':>6} {'PC2%':>6} {'PC3%':>6} {'Top3%':>6} | {'MeanCos':>8}")
    print("-" * 95)

    for ind in sorted(all_deltas):
        deltas = all_deltas[ind]
        n = deltas.shape[0]
        if n < 5:
            continue

        deltas_centered = deltas - deltas.mean(axis=0)
        U, S, Vt = np.linalg.svd(deltas_centered, full_matrices=False)
        var_explained = S**2 / (S**2).sum()

        pc1_dir = Vt[0] / np.linalg.norm(Vt[0])
        cosines = [abs(float(np.dot(d / np.linalg.norm(d), pc1_dir)))
                   for d in deltas if np.linalg.norm(d) > 1e-8]
        mean_cos = np.mean(cosines) if cosines else 0

        print(f"{ind:<45} {n:>4} | {var_explained[0]*100:>5.1f}% {var_explained[1]*100:>5.1f}% "
              f"{var_explained[2]*100:>5.1f}% {sum(var_explained[:3])*100:>5.1f}% | {mean_cos:>8.3f}")

    # Save analysis summary
    summary = {}
    for ind in sorted(all_deltas):
        deltas = all_deltas[ind]
        n = deltas.shape[0]
        if n < 5:
            continue
        deltas_centered = deltas - deltas.mean(axis=0)
        U, S, Vt = np.linalg.svd(deltas_centered, full_matrices=False)
        var_explained = S**2 / (S**2).sum()
        summary[ind] = {
            "n": n,
            "pc1_variance": float(var_explained[0]),
            "pc2_variance": float(var_explained[1]),
            "pc3_variance": float(var_explained[2]),
            "top3_variance": float(sum(var_explained[:3])),
        }

    with open(output_dir / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved activations + PCA summary to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
