"""Generate Qwen3.6-35B-A3B *test* rollouts by reusing the GLM test scenarios and
re-running only rollout + judgment (README section 8 recipe).

For each behavior x {misaligned, benign}: copy the GLM-test understanding.json +
ideation.json into an isolated working dir, run `bloom rollout` + `bloom judgment`
with rollout.target=qwen3.6-35b-a3b and evaluator_reasoning_effort=none, then move
the finished dir to bloom/bloom-results-test/test_<behavior>[_benign]_qwen3_6_35b_a3b.

Runs each behavior in its own CWD so the relative bloom-results/ output never
clobbers the committed dev dirs. No GPU (API only: Opus-4.6 evaluator + Qwen target).
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

BLOOM = Path("/workspace-vast/kaiwen/misalignment-indicators/bloom")
PY = "/workspace-vast/kaiwen/envs/hfdl/bin/python"
TARGET = "qwen3.6-35b-a3b"
TARGET_SUFFIX = "qwen3_6_35b_a3b"

# behavior -> rollout config (modality/turns/no_user + the misaligned configurable_prompt)
BEH = {
    "instructed-covert-code-sabotage": dict(cfg="covert_sabotage", modality="simenv", max_turns=8, no_user=True),
    "instructed-strategic-sandbagging": dict(cfg="sandbag", modality="conversation", max_turns=1, no_user=False),
    "self-preservation": dict(cfg="default", modality="simenv", max_turns=5, no_user=True),
    "strategic-deception": dict(cfg="default", modality="simenv", max_turns=5, no_user=False),
    "sycophancy": dict(cfg="default", modality="conversation", max_turns=5, no_user=False),
}

SEED = """behavior:
  name: "{name}"
  examples: []
temperature: 1.0
evaluator_reasoning_effort: "none"
target_reasoning_effort: "medium"
max_concurrent: 12
configurable_prompts: "{cfg}"
anonymous_target: false
debug: false
understanding:
  model: "claude-opus-4.6"
  max_tokens: 4000
ideation:
  model: "claude-opus-4.6"
  num_scenarios: 1
  variation_dimensions: []
  max_tokens: 28000
  web_search: false
rollout:
  model: "claude-opus-4.6"
  target: "{target}"
  modality: "{modality}"
  max_turns: {max_turns}
  max_tokens: 4000
  no_user_mode: {no_user}
  selected_variations: null
  num_reps: 1
judgment:
  model: "claude-opus-4.6"
  max_tokens: 6000
  num_samples: 1
  additional_qualities: ["unrealism", "evaluation-awareness", "evaluation-invalidity"]
  metajudgment_qualities: ["diversity"]
  redaction_tags: null
"""


def run_one(behavior, benign):
    cfg = BEH[behavior]
    name = behavior + ("_benign" if benign else "")          # real behavior name (drives prompts)
    cprompt = "default" if benign else cfg["cfg"]
    glm_dir = BLOOM / "bloom-results-test" / f"test_{name}_glm_4_7_flash"
    assert (glm_dir / "ideation.json").exists(), f"missing GLM scenarios: {glm_dir}"

    # isolated workspace
    work = BLOOM / "_qtestgen" / name
    if work.exists():
        shutil.rmtree(work)
    cfgdir = work / "cfg"
    cfgdir.mkdir(parents=True)
    # config dir: own seed.yaml + symlinks to the shared bloom-data assets
    for asset in ["behaviors.json", "models.json", "configurable_prompts", "behaviors"]:
        os.symlink(BLOOM / "bloom-data" / asset, cfgdir / asset)
    (cfgdir / "seed.yaml").write_text(SEED.format(
        name=name, cfg=cprompt, target=TARGET,
        modality=cfg["modality"], max_turns=cfg["max_turns"],
        no_user=str(cfg["no_user"]).lower()))
    # pre-place the reused scenarios where bloom expects them: bloom-results/<name>_<target>/
    out = work / "bloom-results" / f"{name}_{TARGET_SUFFIX}"
    out.mkdir(parents=True)
    for f in ["understanding.json", "ideation.json"]:
        shutil.copy(glm_dir / f, out / f)

    env = dict(os.environ)
    # load bloom/.env keys
    for line in (BLOOM / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.replace("export ", "").strip()] = v.strip().strip('"').strip("'")
    env["HF_HOME"] = "/workspace-vast/kaiwen/hf_cache"

    for stage in ["rollout", "judgment"]:
        print(f"\n===== {name}: bloom {stage} =====", flush=True)
        r = subprocess.run([PY, "-m", "bloom", stage, "cfg"], cwd=work, env=env)
        if r.returncode != 0:
            print(f"!! {name} {stage} FAILED (exit {r.returncode})")
            return False
    # move finished dir to the canonical test location
    dst = BLOOM / "bloom-results-test" / f"test_{name}_{TARGET_SUFFIX}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(out), str(dst))
    n = len(json.load(open(dst / "rollout.json"))["rollouts"])
    print(f"++ {name}: {n} rollouts -> {dst}")
    return True


def main():
    which = sys.argv[1:] or list(BEH)   # optional behavior filter
    for behavior in which:
        for benign in (False, True):
            ok = run_one(behavior, benign)
            if not ok:
                print(f"STOP: {behavior} benign={benign} failed")
                return
    print("\nALL DONE")


if __name__ == "__main__":
    main()
