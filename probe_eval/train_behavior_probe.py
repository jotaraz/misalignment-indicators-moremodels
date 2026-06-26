"""
Train a behavior-level linear probe on bloom dev rollouts and evaluate on test.

For self-preferential-bias (single assistant turn per rollout), all reasoning
tokens in a GT-misaligned rollout are labeled positive; all reasoning tokens in
benign/non-misaligned rollouts are labeled negative.

Usage (requires GPU, submit via SLURM):
    python -m probe_eval.train_behavior_probe \
        --behavior self-preferential-bias \
        --layers 27
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ood_misalignment_eval/deception-detection"))

from deception_detection.activations import Activations
from deception_detection.detectors import LogisticRegressionDetector
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import Message

from probe_eval.evaluate import extract_target_dialogue, narrow_detection_mask_to_reasoning

BLOOM_DIR = PROJECT_ROOT / "bloom" / "bloom-results"
BLOOM_TEST_DIR = PROJECT_ROOT / "bloom" / "bloom-results-test"
DEFAULT_OUTPUT = PROJECT_ROOT / "probe_eval" / "results" / "behavior_probe"

TEMPLATE_KWARGS = {"clear_thinking": False}
DEFAULT_PADDING = {k: {"left": 0, "right": 0} for k in ["gemma", "mistral", "llama", "qwen", "glm"]}


def load_gt_misaligned_vars(rollout_dir: Path) -> set[tuple[int, int]]:
    """Load GT and return set of (variation, repetition) that are misaligned."""
    gt_path = rollout_dir / "rollout_misalignment_turns.json"
    if not gt_path.exists():
        return set()
    with open(gt_path) as f:
        gt = json.load(f)
    misaligned = set()
    for r in gt.get("evaluated_rollouts", []):
        if r.get("misaligned_turns"):
            var = r["variation_number"]
            rep = r.get("repetition_number", 1)
            misaligned.add((var, rep))
    return misaligned


def load_rollouts(rollout_dir: Path) -> list[dict]:
    """Load rollouts from a bloom rollout.json file."""
    rollout_path = rollout_dir / "rollout.json"
    if not rollout_path.exists():
        return []
    with open(rollout_path) as f:
        data = json.load(f)
    return data.get("rollouts", [])


def extract_activations_for_rollouts(
    rollouts: list[dict],
    labels: list[bool],
    model,
    tokenizer,
    layers: list[int],
    desc: str = "",
) -> tuple[list[torch.Tensor], list[bool]]:
    """Extract per-token reasoning activations for each rollout.

    Returns:
        per_rollout_acts: list of [n_reasoning_tokens, n_layers, hidden] tensors
        per_rollout_labels: list of bool (same as input labels, minus skipped)
    """
    per_rollout_acts = []
    per_rollout_labels = []
    n_skip = 0

    for i, (rollout, label) in enumerate(zip(rollouts, labels)):
        var = rollout.get("variation_number", 0)
        rep = rollout.get("repetition_number", 1)

        dialogue = extract_target_dialogue(rollout)
        if not dialogue or not any(m.role == "assistant" for m in dialogue):
            n_skip += 1
            continue

        try:
            toks = TokenizedDataset.from_dialogue_list(
                dialogues=[dialogue],
                tokenizer=tokenizer,
                padding=DEFAULT_PADDING,
                template_kwargs=TEMPLATE_KWARGS,
            )

            narrow_detection_mask_to_reasoning(toks, tokenizer)

            acts = Activations.from_model(model, toks, batch_size=1, layers=layers)

            # Get detection mask
            det_mask = toks.detection_mask[0].bool()
            n_det = det_mask.sum().item()

            if n_det == 0:
                n_skip += 1
                continue

            # Extract per-layer activations for detected tokens
            # Activations stores as .all_acts: [batch, seqpos, layer, emb]
            layer_acts = []
            for li, layer in enumerate(layers):
                layer_act = acts.all_acts[0, :, li, :]  # [seq_len, hidden]
                masked = layer_act[det_mask]  # [n_det, hidden]
                layer_acts.append(masked)

            # Stack: [n_det, n_layers, hidden]
            stacked = torch.stack(layer_acts, dim=1).cpu().float()
            per_rollout_acts.append(stacked)
            per_rollout_labels.append(label)

            del acts
            if (i + 1) % 10 == 0:
                print(f"  {desc} [{i+1}/{len(rollouts)}] var={var} n_tokens={n_det} label={'pos' if label else 'neg'}")

        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                print(f"  Skip var={var}: {type(e).__name__}: {str(e)[:100]}")

    print(f"  {desc}: {len(per_rollout_acts)} extracted, {n_skip} skipped")
    return per_rollout_acts, per_rollout_labels


def train_probe(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    layers: list[int],
    reg_coeff: float = 10.0,
    normalize: bool = True,
    lr: float = 1e-2,
    n_steps: int = 500,
) -> LogisticRegressionDetector:
    """Train L2-regularized logistic regression probe. Reuses logic from probe/train.py."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos, n_layers, n_features = pos_acts.shape
    n_neg = neg_acts.shape[0]
    print(f"  Training: {n_pos} pos tokens, {n_neg} neg tokens, {n_layers} layers, {n_features} features")

    X = torch.cat([
        pos_acts.reshape(n_pos, n_layers * n_features),
        neg_acts.reshape(n_neg, n_layers * n_features),
    ], dim=0).float()
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).float()

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

    linear = torch.nn.Linear(n_layers * n_features, 1, bias=False).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(n_steps):
        logits = linear(X).squeeze(-1)
        loss = loss_fn(logits, y)
        l2 = reg_coeff * linear.weight.square().sum() / (2 * len(y))
        (loss + l2).backward()
        optimizer.step()
        optimizer.zero_grad()

    detector = LogisticRegressionDetector(
        layers=layers, reg_coeff=reg_coeff, normalize=normalize
    )
    detector.directions = linear.weight.detach().cpu().reshape(n_layers, n_features)
    detector.scaler_mean = scaler_mean
    detector.scaler_scale = scaler_scale

    return detector


class BilinearProbe(torch.nn.Module):
    """Bilinear probe: score = sum_r (w1_r · x) * (w2_r · x) + w_linear · x.

    Rank controls the number of bilinear interaction terms.
    The linear term preserves the standard probe signal.
    """

    def __init__(self, d: int, rank: int = 2, use_linear: bool = True):
        super().__init__()
        self.w1 = torch.nn.Linear(d, rank, bias=False)
        self.w2 = torch.nn.Linear(d, rank, bias=False)
        self.linear = torch.nn.Linear(d, 1, bias=False) if use_linear else None
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear: sum of (w1_r · x) * (w2_r · x) for each rank
        proj1 = self.w1(x)  # [batch, rank]
        proj2 = self.w2(x)  # [batch, rank]
        bilinear = (proj1 * proj2).sum(dim=-1)  # [batch]
        if self.linear is not None:
            bilinear = bilinear + self.linear(x).squeeze(-1)
        return bilinear


def train_bilinear_probe(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    layers: list[int],
    rank: int = 2,
    reg_coeff: float = 1.0,
    normalize: bool = True,
    lr: float = 1e-3,
    n_steps: int = 1000,
    use_linear: bool = True,
) -> tuple[BilinearProbe, dict]:
    """Train a bilinear probe on per-token activations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos, n_layers, n_features = pos_acts.shape
    n_neg = neg_acts.shape[0]
    d = n_layers * n_features
    print(f"  Bilinear training: {n_pos} pos, {n_neg} neg, d={d}, rank={rank}")

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

    model = BilinearProbe(d, rank=rank, use_linear=use_linear).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(n_steps):
        logits = model(X)
        loss = loss_fn(logits, y)
        # L2 on all parameters
        l2 = reg_coeff * sum(p.square().sum() for p in model.parameters()) / (2 * len(y))
        (loss + l2).backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = (preds == y).float().mean()
                print(f"    step {step+1}: loss={loss.item():.4f} acc={acc.item():.3f}")

    model = model.cpu().eval()
    norm_info = {"scaler_mean": scaler_mean, "scaler_scale": scaler_scale}
    return model, norm_info


def score_rollouts_bilinear(
    per_rollout_acts: list[torch.Tensor],
    model: BilinearProbe,
    norm_info: dict,
    n_layers: int,
) -> list[float]:
    """Score each rollout with a bilinear probe (mean of per-token scores)."""
    scores = []
    scaler_mean = norm_info.get("scaler_mean")
    scaler_scale = norm_info.get("scaler_scale")
    for acts in per_rollout_acts:
        x = acts.reshape(acts.shape[0], -1).float()
        if scaler_mean is not None:
            x = (x - scaler_mean) / scaler_scale
        with torch.no_grad():
            s = model(x)
        scores.append(float(s.mean()))
    return scores


def score_rollouts(
    per_rollout_acts: list[torch.Tensor],
    detector: LogisticRegressionDetector,
) -> list[float]:
    """Score each rollout as mean of per-token probe scores."""
    scores = []
    for acts in per_rollout_acts:
        # acts: [n_tokens, n_layers, hidden]
        score_tensor = detector.get_score_tensor(acts)  # [n_tokens, 1] or [n_tokens]
        if score_tensor.dim() > 1:
            score_tensor = score_tensor.squeeze(-1)
        scores.append(float(score_tensor.mean()))
    return scores


def compute_metrics(scores: list[float], labels: list[bool]) -> dict:
    """Compute AUROC and TPR/FPR at various thresholds."""
    scores_arr = np.array(scores)
    labels_arr = np.array(labels, dtype=int)

    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "n": len(labels_arr), "n_pos": int(n_pos)}

    auroc = float(roc_auc_score(labels_arr, scores_arr))

    # Sweep thresholds
    operating_points = []
    thresholds = np.percentile(scores_arr, np.arange(0, 101, 5))
    thresholds = np.unique(thresholds)
    for t in thresholds:
        preds = scores_arr >= t
        tp = int((preds & labels_arr.astype(bool)).sum())
        fp = int((preds & ~labels_arr.astype(bool)).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        operating_points.append({
            "threshold": float(t),
            "tpr": tpr, "fpr": fpr, "precision": prec,
            "tp": tp, "fp": fp,
        })

    return {
        "auroc": auroc,
        "n": len(labels_arr),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "operating_points": operating_points,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=str, default="self-preferential-bias")
    parser.add_argument("--layers", type=int, nargs="+", default=[27])
    parser.add_argument("--reg-coeff", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--extra-dev-dirs", type=str, nargs="+", default=[],
                        help="Additional dev rollout dirs (with GT). Non-misaligned rollouts are negative.")
    parser.add_argument("--all-positive", action="store_true",
                        help="Treat ALL rollouts in positive dirs as positive (ignore GT).")
    parser.add_argument("--gt-only", action="store_true",
                        help="Only use GT-misaligned from pos dirs as positive, skip non-misaligned (don't add them as negative).")
    parser.add_argument("--probe-type", type=str, default="linear", choices=["linear", "bilinear"],
                        help="Probe architecture: linear or bilinear")
    parser.add_argument("--bilinear-rank", type=int, default=2,
                        help="Rank for bilinear probe (number of interaction terms)")
    parser.add_argument("--no-linear-term", action="store_true",
                        help="For bilinear probe: omit the linear term (pure bilinear)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior = args.behavior

    # --- Resolve rollout dirs ---
    pos_dir_name = f"{behavior}_glm_4_7_flash"
    benign_dir_name = f"{behavior}_benign_glm_4_7_flash"
    test_pos_name = f"test_{behavior}_glm_4_7_flash"
    test_benign_name = f"test_{behavior}_benign_glm_4_7_flash"

    dev_pos_dir = BLOOM_DIR / pos_dir_name
    dev_benign_dir = BLOOM_DIR / benign_dir_name
    test_pos_dir = BLOOM_TEST_DIR / test_pos_name
    test_benign_dir = BLOOM_TEST_DIR / test_benign_name

    # --- Load GT ---
    gt_misaligned = load_gt_misaligned_vars(dev_pos_dir)
    print(f"Dev GT: {len(gt_misaligned)} misaligned rollouts in {pos_dir_name}")

    test_gt_misaligned = load_gt_misaligned_vars(test_pos_dir)
    print(f"Test GT: {len(test_gt_misaligned)} misaligned rollouts in {test_pos_name}")

    # --- Build dev rollout list with labels ---
    dev_pos_rollouts = load_rollouts(dev_pos_dir)
    dev_benign_rollouts = load_rollouts(dev_benign_dir)

    dev_rollouts = []
    dev_labels = []

    for r in dev_pos_rollouts:
        if args.all_positive:
            dev_rollouts.append(r)
            dev_labels.append(True)
        else:
            var = r.get("variation_number", 0)
            rep = r.get("repetition_number", 1)
            is_misaligned = (var, rep) in gt_misaligned
            if args.gt_only and not is_misaligned:
                continue  # skip non-misaligned from pos dir
            dev_rollouts.append(r)
            dev_labels.append(is_misaligned)

    for r in dev_benign_rollouts:
        dev_rollouts.append(r)
        dev_labels.append(False)

    # Extra dev dirs (each has its own GT)
    for extra_dir in args.extra_dev_dirs:
        extra_path = Path(extra_dir)
        extra_rollouts = load_rollouts(extra_path)
        if args.all_positive:
            print(f"Extra dev: {len(extra_rollouts)} rollouts, ALL positive in {extra_path.name}")
            for r in extra_rollouts:
                dev_rollouts.append(r)
                dev_labels.append(True)
        else:
            extra_gt = load_gt_misaligned_vars(extra_path)
            print(f"Extra dev: {len(extra_rollouts)} rollouts, {len(extra_gt)} misaligned in {extra_path.name}")
            for r in extra_rollouts:
                var = r.get("variation_number", 0)
                rep = r.get("repetition_number", 1)
                is_misaligned = (var, rep) in extra_gt
                if args.gt_only and not is_misaligned:
                    continue
                dev_rollouts.append(r)
                dev_labels.append(is_misaligned)

    n_dev_pos = sum(dev_labels)
    n_dev_neg = len(dev_labels) - n_dev_pos
    print(f"Dev set: {n_dev_pos} positive, {n_dev_neg} negative rollouts")

    # --- Build test rollout list ---
    test_pos_rollouts = load_rollouts(test_pos_dir)
    test_benign_rollouts = load_rollouts(test_benign_dir)

    test_rollouts = []
    test_labels = []

    for r in test_pos_rollouts:
        var = r.get("variation_number", 0)
        rep = r.get("repetition_number", 1)
        is_misaligned = (var, rep) in test_gt_misaligned
        test_rollouts.append(r)
        test_labels.append(is_misaligned)

    for r in test_benign_rollouts:
        test_rollouts.append(r)
        test_labels.append(False)

    n_test_pos = sum(test_labels)
    n_test_neg = len(test_labels) - n_test_pos
    print(f"Test set: {n_test_pos} positive, {n_test_neg} negative rollouts")

    # --- Load model ---
    print("\nLoading GLM-4.7 Flash...", flush=True)
    model, tokenizer = get_model_and_tokenizer(ModelName.GLM_FLASH)

    # Warmup
    try:
        warmup = [Message("system", "You are helpful.", False),
                  Message("user", "Hi.", False),
                  Message("assistant", "<think>ok</think>Hello!", True)]
        toks = TokenizedDataset.from_dialogue_list(
            dialogues=[warmup], tokenizer=tokenizer,
            padding=DEFAULT_PADDING, template_kwargs=TEMPLATE_KWARGS)
        Activations.from_model(model, toks, batch_size=1, layers=args.layers)
    except Exception:
        pass

    # --- Extract dev activations ---
    print(f"\n{'='*60}")
    print("Extracting dev activations...")
    print(f"{'='*60}")
    dev_acts, dev_labels_extracted = extract_activations_for_rollouts(
        dev_rollouts, dev_labels, model, tokenizer, args.layers, desc="Dev"
    )

    # --- Train/val split (by rollout) ---
    n_total = len(dev_acts)
    n_val = max(1, int(n_total * args.val_fraction))
    # Stratified: keep ratio of pos/neg
    pos_indices = [i for i, l in enumerate(dev_labels_extracted) if l]
    neg_indices = [i for i, l in enumerate(dev_labels_extracted) if not l]
    np.random.seed(42)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    n_val_pos = max(1, int(len(pos_indices) * args.val_fraction))
    n_val_neg = max(1, int(len(neg_indices) * args.val_fraction))

    val_indices = set(pos_indices[:n_val_pos] + neg_indices[:n_val_neg])
    train_indices = [i for i in range(n_total) if i not in val_indices]

    train_pos_acts = [dev_acts[i] for i in train_indices if dev_labels_extracted[i]]
    train_neg_acts = [dev_acts[i] for i in train_indices if not dev_labels_extracted[i]]
    val_pos_acts = [dev_acts[i] for i in val_indices if dev_labels_extracted[i]]
    val_neg_acts = [dev_acts[i] for i in val_indices if not dev_labels_extracted[i]]

    # Concatenate tokens across rollouts
    train_pos = torch.cat(train_pos_acts, dim=0) if train_pos_acts else torch.zeros(0, len(args.layers), 2048)
    train_neg = torch.cat(train_neg_acts, dim=0) if train_neg_acts else torch.zeros(0, len(args.layers), 2048)

    print(f"\nTrain: {len(train_indices)} rollouts ({len(train_pos_acts)} pos, {len(train_neg_acts)} neg)")
    print(f"  Tokens: {train_pos.shape[0]} pos, {train_neg.shape[0]} neg")
    print(f"Val: {len(val_indices)} rollouts ({len(val_pos_acts)} pos, {len(val_neg_acts)} neg)")

    # --- Train probe ---
    print(f"\n{'='*60}")
    print(f"Training {args.probe_type} probe...")
    print(f"{'='*60}")

    bilinear_model = None
    bilinear_norm = None
    detector = None

    if args.probe_type == "bilinear":
        bilinear_model, bilinear_norm = train_bilinear_probe(
            train_pos, train_neg, args.layers,
            rank=args.bilinear_rank,
            reg_coeff=args.reg_coeff,
            lr=1e-3, n_steps=args.n_steps,
            use_linear=not args.no_linear_term,
        )
    else:
        detector = train_probe(
            train_pos, train_neg, args.layers,
            reg_coeff=args.reg_coeff, n_steps=args.n_steps,
        )

    # --- Val metrics ---
    if val_pos_acts and val_neg_acts:
        if args.probe_type == "bilinear":
            val_scores = score_rollouts_bilinear(
                val_pos_acts + val_neg_acts, bilinear_model, bilinear_norm, len(args.layers))
        else:
            val_scores = score_rollouts(val_pos_acts + val_neg_acts, detector)
        val_labels = [True] * len(val_pos_acts) + [False] * len(val_neg_acts)
        val_metrics = compute_metrics(val_scores, val_labels)
        print(f"\nVal AUROC: {val_metrics['auroc']:.4f}" if val_metrics['auroc'] else "\nVal: insufficient data")

    # --- Save probe ---
    layer_str = f"layer{args.layers[0]}" if len(args.layers) == 1 else f"layers{'_'.join(map(str, args.layers))}"
    probe_type_suffix = f"_{args.probe_type}" if args.probe_type != "linear" else ""
    probe_dir = output_dir / f"{behavior}_behavior{probe_type_suffix}" / "turn" / layer_str
    probe_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_type == "bilinear":
        torch.save({"model": bilinear_model.state_dict(), "norm_info": bilinear_norm,
                     "rank": args.bilinear_rank, "use_linear": not args.no_linear_term,
                     "d": train_pos.shape[1] * train_pos.shape[2]},
                    probe_dir / "bilinear_probe.pt")
    else:
        detector.save(probe_dir / "detector.pt")

    cfg_dict = {
        "method": "lr",
        "model_name": "glm-9b-flash",
        "train_data": "repe_honesty__you_are_fact_sys",
        "eval_data": [],
        "control_data": [],
        "trim_reasoning": True,
        "train_on_policy": False,
        "eval_on_policy": True,
        "control_on_policy": False,
        "detect_only_start_of_turn": False,
        "detect_only_last_token": False,
        "val_fraction": 0.0,
        "control_dataset_size": 0,
        "detect_layers": args.layers,
        "detect_num_latents": None,
        "use_local_sae_acts": False,
        "use_goodfire_sae_acts": False,
        "sae_latent_whitelist": None,
        "pw_locked": False,
        "lora_path": None,
        "reg_coeff": args.reg_coeff,
        "normalize_acts": True,
        "max_llama_token_length": None,
        "use_followup_question": False,
        "id": f"{behavior}_behavior",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(probe_dir / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, indent=2)

    meta = {
        "behavior": behavior,
        "probe_type": args.probe_type,
        "label_mode": "behavior_level",
        "layers": args.layers,
        "reg_coeff": args.reg_coeff,
        "bilinear_rank": args.bilinear_rank if args.probe_type == "bilinear" else None,
        "n_train_pos_rollouts": len(train_pos_acts),
        "n_train_neg_rollouts": len(train_neg_acts),
        "n_train_pos_tokens": int(train_pos.shape[0]),
        "n_train_neg_tokens": int(train_neg.shape[0]),
        "n_val_pos_rollouts": len(val_pos_acts),
        "n_val_neg_rollouts": len(val_neg_acts),
    }
    if val_pos_acts and val_neg_acts:
        meta["val_auroc"] = val_metrics.get("auroc")
    with open(probe_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nProbe saved to {probe_dir}")

    # --- Evaluate on test ---
    print(f"\n{'='*60}")
    print("Extracting test activations...")
    print(f"{'='*60}")
    test_acts, test_labels_extracted = extract_activations_for_rollouts(
        test_rollouts, test_labels, model, tokenizer, args.layers, desc="Test"
    )

    if args.probe_type == "bilinear":
        test_scores = score_rollouts_bilinear(
            test_acts, bilinear_model, bilinear_norm, len(args.layers))
    else:
        test_scores = score_rollouts(test_acts, detector)
    test_metrics = compute_metrics(test_scores, test_labels_extracted)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS — {behavior} behavior probe")
    print(f"{'='*60}")
    print(f"AUROC: {test_metrics['auroc']:.4f}" if test_metrics['auroc'] else "AUROC: N/A")
    print(f"N: {test_metrics['n']} ({test_metrics.get('n_pos', 0)} pos, {test_metrics.get('n_neg', 0)} neg)")

    if test_metrics.get("operating_points"):
        print(f"\n{'Threshold':>10} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'TP':>4} {'FP':>4}")
        print("-" * 50)
        for op in test_metrics["operating_points"]:
            print(f"{op['threshold']:>10.4f} {op['tpr']:>7.1%} {op['fpr']:>7.1%} "
                  f"{op['precision']:>7.1%} {op['tp']:>4} {op['fp']:>4}")

    # Save test results
    results_path = output_dir / f"{behavior}_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "behavior": behavior,
            "layers": args.layers,
            "test_metrics": test_metrics,
            "test_scores": [{"score": s, "label": l} for s, l in zip(test_scores, test_labels_extracted)],
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
