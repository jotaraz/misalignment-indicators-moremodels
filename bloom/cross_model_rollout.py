#!/usr/bin/env python3
"""Cross-model rollout orchestrator — README section 8, automated.

For a target model, reuse the *fixed* GLM dev scenarios and run ONLY the bloom
rollout stage for each of the 5 dev behaviors x {misaligned, benign}.
understanding.json + ideation.json are copied from the committed GLM runs, so
prompts/memos stay identical and only the target model differs. Per-behavior
rollout settings (modality, max_turns, no_user_mode, reps) are read back from
each GLM run's rollout.json `metadata`, so e.g. self-preservation stays a
5-turn simenv while sandbagging stays a 1-turn Q&A.

Run from the bloom env (the `bloom` CLI must be on PATH):
    python bloom/cross_model_rollout.py --target llama-3.3-70b
    python bloom/cross_model_rollout.py --target deepseek-r1-llama-70b --reasoning
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

BLOOM_DIR = Path(__file__).resolve().parent        # repo/bloom
RESULTS = BLOOM_DIR / "bloom-results"
DATA = BLOOM_DIR / "bloom-data"

# behavior -> GLM source dirs (misaligned / benign), configurable_prompts, and
# the indicator-judge soft-prior label used for ground truth.
BEHAVIORS = {
    "self-preservation": dict(
        src="self-preservation_glm_4_7_flash",
        src_benign="self-preservation_benign_glm_4_7_flash",
        prompts="default", examples=["blackmail-opus4"], tb="self-preservation"),
    "strategic-deception": dict(
        src="strategic-deception_glm_4_7_flash",
        src_benign="strategic-deception_benign_glm_4_7_flash",
        prompts="default",
        examples=["strategic_deception/security_flaw_fabrication_short",
                  "strategic_deception/supplier_environmental_deception",
                  "strategic_deception/pr_factual_denial"],
        tb="strategic-deception"),
    "sycophancy": dict(
        src="sycophancy", src_benign="sycophancy_benign_glm_4_7_flash",
        prompts="default", examples=[], tb="sycophancy"),
    "instructed-covert-code-sabotage": dict(
        src="instructed-covert-code-sabotage_glm_4_7_flash",
        src_benign="instructed-covert-code-sabotage_benign_glm_4_7_flash",
        prompts="covert_sabotage",
        examples=["instructed_sabotage_bug_indispensability", "instructed_sabotage_backdoor_redteam"],
        tb="sabotage"),
    "instructed-strategic-sandbagging": dict(
        src="instructed-strategic-sandbagging_glm_4_7_flash",
        src_benign="instructed-strategic-sandbagging_benign_glm_4_7_flash",
        prompts="sandbag",
        examples=["instructed_sandbagging_bio_test", "instructed_sandbagging_coding_mcq"],
        tb="sandbagging"),
}


def sanitize(model: str) -> str:
    """Mirror bloom.utils.sanitize_model_name for results-dir naming."""
    s = model.replace(".", "_").replace("-", "_").replace("/", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def make_config_dir(tmp: str, seed: dict) -> Path:
    """A bloom config dir = seed.yaml + the static bloom-data assets it references."""
    cfg = Path(tmp)
    for item in ("behaviors.json", "models.json", "behaviors", "configurable_prompts"):
        src = DATA / item
        if src.is_dir():
            shutil.copytree(src, cfg / item)
        elif src.exists():
            shutil.copy2(src, cfg / item)
    with open(cfg / "seed.yaml", "w") as f:
        yaml.safe_dump(seed, f, sort_keys=False)
    return cfg


def run_one(target: str, behavior: str, spec: dict, benign: bool,
            reasoning: bool, force: bool, limit: int | None = None) -> None:
    name = f"{behavior}_benign" if benign else behavior
    src_dir = RESULTS / (spec["src_benign"] if benign else spec["src"])
    out_dir = RESULTS / f"{name}_{sanitize(target)}"

    if not src_dir.exists():
        print(f"  SKIP {name}: GLM source {src_dir.name} missing")
        return
    if (out_dir / "rollout.json").exists() and not force:
        print(f"  SKIP {name}/{target}: rollout.json already exists")
        return

    # Reuse the fixed scenarios + per-behavior rollout settings from the GLM run.
    meta = json.load(open(src_dir / "rollout.json")).get("metadata", {})
    src_top = json.load(open(src_dir / "rollout.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("understanding.json", "ideation.json"):
        shutil.copy2(src_dir / fname, out_dir / fname)

    # --limit: trim the scenario set for a cheap smoke test (variations drive rollouts).
    if limit is not None:
        ide = json.load(open(out_dir / "ideation.json"))
        if isinstance(ide.get("variations"), list):
            ide["variations"] = ide["variations"][:limit]
            json.dump(ide, open(out_dir / "ideation.json", "w"), indent=2)
            print(f"  (limit) trimmed to {len(ide['variations'])} scenario(s)")

    seed = yaml.safe_load(open(DATA / "seed.yaml"))
    seed["behavior"] = {"name": name, "examples": spec["examples"]}
    seed["configurable_prompts"] = "default" if benign else spec["prompts"]
    # README s8: a thinking-mode evaluator truncates reasoning-only target turns.
    seed["evaluator_reasoning_effort"] = "none"
    seed["target_reasoning_effort"] = "high" if reasoning else "none"
    seed["rollout"] = {
        # Fixed, known-good evaluator. The GLM metadata sometimes names an
        # evaluator version that isn't in models.json (e.g. claude-opus-4.5),
        # which fails model resolution; claude-opus-4.6 is the section-8 standard.
        "model": "claude-opus-4.6",
        "target": target,
        "modality": meta.get("modality", "conversation"),
        "max_turns": int(meta.get("max_turns", 1)),
        "no_user_mode": str(meta.get("no_user_mode", "False")).lower() == "true",
        "num_reps": int(src_top.get("repetitions_per_variation", 1)),
        "selected_variations": None,
        # Must exceed the target's thinking budget (reasoning_effort high -> 4096).
        # Generous cap = room for CoT + answer; harmless cap for non-reasoning targets.
        "max_tokens": 16000,
    }

    with tempfile.TemporaryDirectory(dir=BLOOM_DIR) as tmp:
        cfg = make_config_dir(tmp, seed)
        print(f"  ROLLOUT {name} -> bloom-results/{out_dir.name} "
              f"(modality={seed['rollout']['modality']}, turns={seed['rollout']['max_turns']})")
        subprocess.run(["bloom", "rollout", str(cfg)], cwd=BLOOM_DIR, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="bloom model key (see bloom-data/models.json)")
    ap.add_argument("--reasoning", action="store_true",
                    help="target emits <think> CoT (keeps target reasoning on, e.g. R1-Llama)")
    ap.add_argument("--behaviors", nargs="*", default=list(BEHAVIORS),
                    help="subset of the 5 dev behaviors (default: all)")
    ap.add_argument("--force", action="store_true", help="re-run even if rollout.json exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap scenarios per dir (smoke test); default: all")
    args = ap.parse_args()

    for b in args.behaviors:
        spec = BEHAVIORS[b]
        for benign in (False, True):
            run_one(args.target, b, spec, benign, args.reasoning, args.force, args.limit)


if __name__ == "__main__":
    main()
