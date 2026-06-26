"""Ground-truth labeling for the Qwen3.6-35B-A3B test rollouts (README section 8 step 4).

Misaligned dirs: Opus-4.6 misalignment-turn judge (defines GT positive turns) +
ambiguous audit. Benign dirs: a minimal GT is constructed from rollout.json (all
turns negative) -- sufficient for the probe AUROC negative pool. The per-indicator
judge (--indicator-turns) and benign indicator pass are NOT run here; they feed the
LLM-judge baseline, not the probe misalignment AUROC.

Run from bloom/ with the hfdl env. API only (Opus-4.6).
"""
import json, os, subprocess, sys
from pathlib import Path

BLOOM = Path("/workspace-vast/kaiwen/misalignment-indicators/bloom")
REPO = BLOOM.parent
PY = "/workspace-vast/kaiwen/envs/hfdl/bin/python"
SUFFIX = "qwen3_6_35b_a3b"
TESTBEH = {
    "instructed-covert-code-sabotage": "sabotage",
    "instructed-strategic-sandbagging": "sandbagging",
    "self-preservation": "self-preservation",
    "strategic-deception": "strategic-deception",
    "sycophancy": "sycophancy",
}


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO)
    for line in (BLOOM / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            e[k.replace("export ", "").strip()] = v.strip().strip('"').strip("'")
    return e


def gt_misaligned(behavior):
    d = BLOOM / "bloom-results-test" / f"test_{behavior}_{SUFFIX}"
    rj = d / "rollout.json"
    out = d / "rollout_misalignment_turns.json"
    assert rj.exists(), f"no rollout.json in {d}"
    print(f"\n=== {behavior}: misalignment-turn judge (Opus-4.6) ===", flush=True)
    r = subprocess.run([PY, "-m", "black_box_ind_judge", str(rj), "--misalignment-turns",
                        "-p", "anthropic", "-m", "claude-opus-4-6", "--thinking-budget", "20000",
                        "-o", str(out)], cwd=REPO, env=env())
    if r.returncode != 0:
        print(f"!! {behavior} misalignment-turns FAILED"); return False
    print(f"=== {behavior}: ambiguous audit ===", flush=True)
    subprocess.run([PY, "-m", "black_box_ind_judge.audit",
                    "--source", f"bloom-results-test/test_{behavior}_{SUFFIX}",
                    "--model", "claude-opus-4-6"], cwd=REPO, env=env())
    n = sum(1 for x in json.load(open(out))["evaluated_rollouts"] if x.get("misaligned_turns"))
    tot = len(json.load(open(out))["evaluated_rollouts"])
    print(f"++ {behavior}: {n}/{tot} rollouts have misaligned turns")
    return True


def gt_benign(behavior):
    """Construct a minimal benign GT (all turns negative) from rollout.json."""
    d = BLOOM / "bloom-results-test" / f"test_{behavior}_benign_{SUFFIX}"
    rj = json.load(open(d / "rollout.json"))
    ev = [{"variation_number": x["variation_number"],
           "repetition_number": x["repetition_number"],
           "misaligned_turns": []} for x in rj["rollouts"]]
    out = {"metadata": {"judge_type": "benign_constructed", "target": "qwen3.6-35b-a3b",
                        "note": "benign control: all turns negative by construction"},
           "evaluated_rollouts": ev}
    json.dump(out, open(d / "rollout_misalignment_turns.json", "w"), indent=1)
    print(f"++ {behavior}_benign: minimal GT for {len(ev)} rollouts")


def main():
    which = sys.argv[1:] or list(TESTBEH)
    for behavior in which:
        if not gt_misaligned(behavior):
            print(f"STOP at {behavior}"); return
        gt_benign(behavior)
    print("\nGT DONE")


if __name__ == "__main__":
    main()
