"""
Extract and save GLM-4.7 Flash activations for bloom dev/test rollouts.

Saves per-rollout activations as .npz files for reuse across probe evaluations,
avoiding repeated GPU forward passes. Also saves metadata (dialogue info, labels,
turn maps) for downstream analysis.

Output structure:
    {output_dir}/{behavior}/
        activations_{var}_{rep}.npz   — per-token activations at specified layers
        metadata.json                 — rollout metadata, labels, turn maps

Usage (requires GPU):
    python -m probe_eval.extract_bloom_activations --split dev
    python -m probe_eval.extract_bloom_activations --split test
    python -m probe_eval.extract_bloom_activations --split dev --layers 27 --batch-size 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ood_misalignment_eval/deception-detection"))

from probe_eval.evaluate import extract_target_dialogue

# Bloom rollout directories
DEV_POSITIVE = [
    "instructed-long-horizon-sabotage-backdoor",
    "instructed-long-horizon-sabotage-bug",
    "sycophancy",
    "strategic-deception_glmflash",
    "self-preferential-bias_glm_4_7_flash",
    "self-promotion_glm_4_7_flash",
    "self-preservation_glm_4_7_flash",
]
DEV_BENIGN = [
    "instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash",
    "instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash",
    "sycophancy_benign_glm_4_7_flash",
    "strategic-deception_benign_glm_4_7_flash",
    "self-preferential-bias_benign_glm_4_7_flash",
    "self-promotion_benign_glm_4_7_flash",
]
TEST_POSITIVE = [
    "test_instructed-long-horizon-sabotage-backdoor__glm_4_7_flash",
    "test_instructed-long-horizon-sabotage-bug_glm_4_7_flash",
    "test_sycophancy_glm_4_7_flash",
    "test_strategic-deception_glm_4_7_flash",
    "test_self-preferential-bias_glm_4_7_flash",
    "test_self-promotion_glm_4_7_flash",
    "test_self-preservation_glm_4_7_flash",
    "test_self-preservation_benign_glm_4_7_flash",
]
TEST_BENIGN = [
    "test_instructed-long-horizon-sabotage-backdoor_benign_glm_4_7_flash",
    "test_instructed-long-horizon-sabotage-bug_benign_glm_4_7_flash",
    "test_sycophancy_benign_glm_4_7_flash",
    "test_strategic-deception_benign_glm_4_7_flash",
    "test_self-preferential-bias_benign_glm_4_7_flash",
    "test_self-promotion_benign_glm_4_7_flash",
]
# OOD eval rollout directories (stored under ood_misalignment_eval/)
OOD_DIRS = [
    "agentic-misalignment/results/ood_eval_glm/bloom_rollout",
    "deception-bench/rollouts/glm-4.7-flash/bloom",
    "sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_answer",
    "sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_are_you_sure",
    "sycophancy-eval/rollouts/glm-4.7-flash/bloom/sycophancy_feedback",
    "mask/rollouts/glm-4.7-flash/mask",
]


def extract_behavior_activations(
    behavior_dir: Path,
    model,
    tokenizer,
    layers: list[int],
    output_dir: Path,
    reasoning_only: bool = True,
) -> dict:
    """Extract activations for all rollouts in a behavior directory."""
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.activations import Activations

    rollout_path = behavior_dir / "rollout.json"
    if not rollout_path.exists():
        print(f"  SKIP: {rollout_path} not found")
        return {}

    with open(rollout_path) as f:
        data = json.load(f)

    behavior_name = behavior_dir.name
    out_dir = output_dir / behavior_name
    out_dir.mkdir(parents=True, exist_ok=True)

    template_kwargs = {"clear_thinking": False}
    default_padding = {k: {"left": 0, "right": 0} for k in ["gemma", "mistral", "llama", "qwen", "glm"]}

    metadata = {
        "behavior": behavior_name,
        "model": str(getattr(tokenizer, "name_or_path", "")),
        "layers": layers,
        "rollouts": [],
    }

    n_saved = 0
    n_skipped = 0

    for rollout in data.get("rollouts", []):
        var = rollout.get("variation_number", 0)
        rep = rollout.get("repetition_number", 1)

        # Check if already extracted
        act_path = out_dir / f"activations_{var}_{rep}.npz"
        if act_path.exists():
            n_saved += 1
            continue

        dialogue, turn_map = extract_target_dialogue(rollout, return_turn_map=True)
        if not dialogue or not any(m.role == "assistant" for m in dialogue):
            n_skipped += 1
            continue

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=[dialogue],
                tokenizer=tokenizer,
                padding=default_padding,
                template_kwargs=template_kwargs,
            )

            if reasoning_only:
                from probe_eval.evaluate import narrow_detection_mask_to_reasoning
                narrow_detection_mask_to_reasoning(toks, tokenizer)

            acts = Activations.from_model(model, toks, batch_size=1, layers=layers)

            # Save per-layer masked activations (detection tokens only)
            det_mask = toks.detection_mask[0].bool() if toks.detection_mask is not None else toks.attention_mask[0].bool()

            save_dict = {}
            for li, layer in enumerate(layers):
                # acts.activations shape: [1, seq_len, n_layers, hidden_dim]
                layer_acts = acts.activations[0, :, li, :]  # [seq_len, hidden_dim]
                masked_acts = layer_acts[det_mask].cpu().float().numpy()
                save_dict[f"layer{layer}"] = masked_acts

            # Also save token info
            attn_mask = toks.attention_mask[0].bool()
            token_ids = toks.tokens[0][attn_mask]
            det_within_attn = det_mask[attn_mask]

            np.savez_compressed(
                act_path,
                **save_dict,
                detection_mask=det_within_attn.numpy(),
            )

            metadata["rollouts"].append({
                "variation": var,
                "repetition": rep,
                "n_tokens": int(attn_mask.sum()),
                "n_detected": int(det_mask.sum()),
                "turn_map": turn_map,
            })

            n_saved += 1
            del acts

        except Exception as e:
            n_skipped += 1
            if n_skipped <= 3:
                print(f"    Skip var={var}: {type(e).__name__}: {str(e)[:100]}")

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {behavior_name}: {n_saved} saved, {n_skipped} skipped")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--layers", type=int, nargs="+", default=[27, 28, 29, 30])
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "probe_eval/cached_activations"))
    parser.add_argument("--reasoning-only", action="store_true", default=True)
    parser.add_argument("--bloom-dir", type=str,
                        default=str(PROJECT_ROOT / "bloom/bloom-results"))
    parser.add_argument("--bloom-test-dir", type=str,
                        default=str(PROJECT_ROOT / "bloom/bloom-results-test"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bloom_dir = Path(args.bloom_dir)
    bloom_test_dir = Path(args.bloom_test_dir)

    # Load model
    from deception_detection.models import get_model_and_tokenizer, ModelName
    print("Loading GLM-4.7 Flash...", flush=True)
    model, tokenizer = get_model_and_tokenizer(ModelName.GLM_FLASH)

    # Warmup
    print("Warmup pass...", flush=True)
    try:
        from deception_detection.tokenized_data import TokenizedDataset
        from deception_detection.activations import Activations
        from deception_detection.utils import Message
        warmup_dialogue = [
            Message("system", "You are a helpful assistant.", False),
            Message("user", "Hello.", False),
            Message("assistant", "<think>Hi</think>Hello!", True),
        ]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues=[warmup_dialogue], tokenizer=tokenizer,
            padding={k: {"left": 0, "right": 0} for k in ["gemma", "mistral", "llama", "qwen", "glm"]},
            template_kwargs={"clear_thinking": False},
        )
        Activations.from_model(model, toks, batch_size=1, layers=args.layers)
    except Exception:
        pass

    ood_dir = PROJECT_ROOT / "ood_misalignment_eval"

    # Build behavior list
    behaviors = []
    if args.split in ("dev", "both"):
        for b in DEV_POSITIVE + DEV_BENIGN:
            behaviors.append(("dev", bloom_dir / b))
    if args.split in ("test", "both"):
        for b in TEST_POSITIVE + TEST_BENIGN:
            behaviors.append(("test", bloom_test_dir / b))
        # OOD evals go in test
        for b in OOD_DIRS:
            behaviors.append(("test", ood_dir / b))

    print(f"\nExtracting activations for {len(behaviors)} behavior directories", flush=True)
    print(f"Layers: {args.layers}, Output: {output_dir}", flush=True)

    for split, bdir in behaviors:
        if not bdir.exists():
            print(f"  SKIP: {bdir} not found")
            continue
        extract_behavior_activations(
            bdir, model, tokenizer, args.layers,
            output_dir / split, args.reasoning_only,
        )

    print(f"\nDone. Activations saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
