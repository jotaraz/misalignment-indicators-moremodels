"""Swap 3 CLEAR (dev → test) with 3 BENIGN (test → dev).

Dev:  13 CLEAR + 7 BENIGN  →  10 CLEAR + 10 BENIGN
Test: 11 CLEAR + 27 BENIGN + 2 AMBIG  →  14 CLEAR + 24 BENIGN + 2 AMBIG

Picks:
- 3 CLEAR from dev preferring newer-regime (v2) so test gets fresh CLEAR
- 3 BENIGN from test preferring older-regime (test_v1) to balance dev's regime mix

All 5 files reindexed + transcript files renamed. Cross-references updated.
"""
import json, shutil, random
from pathlib import Path

random.seed(7)

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/bloom")
DEV = ROOT / "bloom-results/strategic-deception_glm_4_7_flash"
TEST = ROOT / "bloom-results-test/test_strategic-deception_glm_4_7_flash"

def load_dir(d: Path):
    rollout = json.load(open(d / "rollout.json"))
    ideation = json.load(open(d / "ideation.json"))
    mtj = json.load(open(d / "rollout_misalignment_turns.json"))
    audit = json.load(open(d / "ambiguous_rollouts.json"))
    judgment = json.load(open(d / "judgment.json"))
    mapping = json.load(open(d / "combination_mapping.json"))
    return rollout, ideation, mtj, audit, judgment, mapping

def items_for_dir(d: Path):
    rollout, ideation, mtj, audit, judgment, mapping = load_dir(d)
    mtj_by_var = {r["variation_number"]: r for r in mtj["evaluated_rollouts"]}
    audit_by_var = {r["variation_number"]: r for r in audit["audit_details"]}
    judg_by_var = {r["variation_number"]: r for r in judgment["judgments"]}
    map_by_var = {m["new_variation_number"]: m for m in mapping["mapping"]}
    items = []
    for ro in rollout["rollouts"]:
        v = ro["variation_number"]
        items.append({
            "src_dir": d,
            "var": v,
            "rep": ro["repetition_number"],
            "rollout": ro,
            "ideation_entry": ideation["variations"][v-1],
            "mtj_entry": mtj_by_var.get(v),
            "audit_entry": audit_by_var.get(v),
            "judgment_entry": judg_by_var.get(v),
            "map_entry": map_by_var.get(v),
        })
    return items, ideation, mtj, audit, judgment, mapping

def cls_of(it):
    if it["audit_entry"] is None or not (it["mtj_entry"] and it["mtj_entry"].get("misaligned_turns")):
        return "BENIGN"
    return it["audit_entry"]["verdict"]  # CLEAR or AMBIGUOUS

# Load both
print("Loading dev and test...")
dev_items, dev_ide, dev_mtj, dev_audit, dev_judg, dev_map = items_for_dir(DEV)
test_items, test_ide, test_mtj, test_audit, test_judg, test_map = items_for_dir(TEST)

print(f"Dev: {len(dev_items)}  Test: {len(test_items)}")

# Pick 3 CLEAR from dev — prefer ones whose original src_dir is the new v2
dev_clear = [it for it in dev_items if cls_of(it) == "CLEAR"]
print(f"Dev CLEAR pool: {len(dev_clear)}")
dev_clear_sorted = sorted(dev_clear, key=lambda it: 0 if "_v2" in it["map_entry"]["src_dir"] else 1)
dev_to_move = dev_clear_sorted[:3]
print("Picking from dev (3 CLEAR):")
for it in dev_to_move:
    print(f"  new_var={it['var']} src={it['map_entry']['src_dir']} src_var={it['map_entry']['src_variation_number']}")

# Pick 3 BENIGN from test — prefer test_v1 source
test_benign = [it for it in test_items if cls_of(it) == "BENIGN"]
print(f"Test BENIGN pool: {len(test_benign)}")
test_benign_sorted = sorted(test_benign, key=lambda it: 0 if "test_strategic-deception_glm_4_7_flash_v1" in it["map_entry"]["src_dir"] else 1)
test_to_move = test_benign_sorted[:3]
print("Picking from test (3 BENIGN):")
for it in test_to_move:
    print(f"  new_var={it['var']} src={it['map_entry']['src_dir']} src_var={it['map_entry']['src_variation_number']}")

dev_to_move_vars = set(it["var"] for it in dev_to_move)
test_to_move_vars = set(it["var"] for it in test_to_move)

# Construct new dev = (dev keep) + test_to_move
new_dev_items = [it for it in dev_items if it["var"] not in dev_to_move_vars] + test_to_move
# Construct new test = (test keep) + dev_to_move
new_test_items = [it for it in test_items if it["var"] not in test_to_move_vars] + dev_to_move

print(f"\nNew dev size: {len(new_dev_items)}  New test size: {len(new_test_items)}")
assert len(new_dev_items) == 20
assert len(new_test_items) == 40

def write_dir(out_dir: Path, items: list[dict], src_judg_template: dict, src_mtj_meta: dict, behavior_name: str):
    """Reindex items 1..N, copy/rename transcripts, write all 5 files + mapping."""
    # Old transcripts to delete (only those we're removing — but easier: delete all transcripts then recopy)
    for f in out_dir.glob("transcript_v*r1.json"):
        f.unlink()

    new_rollouts, new_variations, new_mtj, new_audit_details, new_judgments = [], [], [], [], []
    new_mapping = []
    audited_count = 0
    num_clear = 0
    num_ambig = 0
    misaligned_count = 0

    for new_idx, it in enumerate(items, start=1):
        ro = dict(it["rollout"])
        ro["variation_number"] = new_idx
        ro["variation_description"] = it["ideation_entry"]["description"]
        new_rollouts.append(ro)
        new_variations.append({"description": it["ideation_entry"]["description"], "tools": it["ideation_entry"].get("tools", [])})
        if it["mtj_entry"] is not None:
            m = dict(it["mtj_entry"]); m["variation_number"] = new_idx
            new_mtj.append(m)
            if m.get("misaligned_turns"):
                misaligned_count += 1
        if it["audit_entry"] is not None:
            a = dict(it["audit_entry"]); a["variation_number"] = new_idx
            new_audit_details.append(a)
            audited_count += 1
            if a["verdict"] == "CLEAR": num_clear += 1
            elif a["verdict"] == "AMBIGUOUS": num_ambig += 1
        if it["judgment_entry"] is not None:
            j = dict(it["judgment_entry"]); j["variation_number"] = new_idx
            new_judgments.append(j)
        # Copy transcript file from source dir (it["src_dir"] is dev or test; original file name uses old var)
        src_transcript = it["src_dir"] / f"transcript_v{it['var']}r{it['rep']}.json"
        dst_transcript = out_dir / f"transcript_v{new_idx}r{it['rep']}.json"
        if src_transcript.exists():
            shutil.copy(src_transcript, dst_transcript)
        # Mapping: preserve original chain back to absolute source
        orig_src_dir = it["map_entry"]["src_dir"] if it["map_entry"] else "unknown"
        orig_src_var = it["map_entry"]["src_variation_number"] if it["map_entry"] else it["var"]
        new_mapping.append({
            "new_variation_number": new_idx,
            "src_dir": orig_src_dir,
            "src_variation_number": orig_src_var,
            "src_repetition_number": it["rep"],
            "classification": cls_of(it),
        })

    n = len(items)
    json.dump({
        "metadata": {"behavior": behavior_name, "combined_from": sorted(set(m["src_dir"] for m in new_mapping))},
        "rollouts": new_rollouts,
        "successful_count": n, "failed_count": 0, "total_count": n,
        "variations_count": n, "repetitions_per_variation": 1,
    }, open(out_dir / "rollout.json", "w"), indent=2)

    json.dump({
        "behavior_name": behavior_name, "examples": [], "model": "combined",
        "num_scenarios": n, "variation_dimensions": [], "total_evals": n,
        "variations": new_variations,
    }, open(out_dir / "ideation.json", "w"), indent=2)

    json.dump({
        "metadata": dict(src_mtj_meta),
        "evaluated_rollouts": new_mtj,
    }, open(out_dir / "rollout_misalignment_turns.json", "w"), indent=2)

    json.dump({
        "behavior": behavior_name, "dir": out_dir.name, "model": "combined",
        "total_rollouts": n, "misaligned_rollouts": misaligned_count,
        "audited": audited_count, "num_ambiguous": num_ambig, "num_clear": num_clear,
        "ambiguous_rollouts": [a for a in new_audit_details if a["verdict"] == "AMBIGUOUS"],
        "audit_details": new_audit_details,
    }, open(out_dir / "ambiguous_rollouts.json", "w"), indent=2)

    new_judg = dict(src_judg_template)
    new_judg["judgments"] = new_judgments
    new_judg["total_conversations"] = n
    new_judg["successful_count"] = len(new_judgments)
    new_judg["failed_count"] = n - len(new_judgments)
    json.dump(new_judg, open(out_dir / "judgment.json", "w"), indent=2)

    json.dump({"mapping": new_mapping, "selection_summary": {"total": n, "clear": num_clear, "ambig": num_ambig, "benign": n - audited_count}},
              open(out_dir / "combination_mapping.json", "w"), indent=2)

# Use existing dev files as templates (judgment + mtj metadata)
write_dir(DEV, new_dev_items, dev_judg, dev_mtj["metadata"], "strategic-deception")
write_dir(TEST, new_test_items, test_judg, test_mtj["metadata"], "strategic-deception")

print("\n=== Final ===")
for label, d in [("DEV", DEV), ("TEST", TEST)]:
    audit = json.load(open(d / "ambiguous_rollouts.json"))
    rollout = json.load(open(d / "rollout.json"))
    n = len(rollout["rollouts"])
    benign = n - audit["audited"]
    print(f"{label}: total={n} CLEAR={audit['num_clear']} AMBIG={audit['num_ambiguous']} BENIGN={benign}")
