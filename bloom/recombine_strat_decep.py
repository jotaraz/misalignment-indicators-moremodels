"""Recombine strategic-deception: 10 CLEAR for dev, 14 CLEAR for test. No ambiguous, no benign.

Dev (10): 8 from v1_dev + 2 from v2_dev
Test (14): 9 from test_v1 + 4 from v2_dev + 1 from v3_dev
"""
import json, shutil
from pathlib import Path

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/bloom")

def load_clear_items(d: Path):
    """Load all CLEAR rollouts from a dir, returning (item_dict, source_dir)."""
    rollout = json.load(open(d / "rollout.json"))
    ideation = json.load(open(d / "ideation.json"))
    mtj = json.load(open(d / "rollout_misalignment_turns.json"))
    audit = json.load(open(d / "ambiguous_rollouts.json"))
    judgment = json.load(open(d / "judgment.json"))

    mtj_by_key = {(r["variation_number"], r["repetition_number"]): r for r in mtj["evaluated_rollouts"]}
    audit_by_key = {(x["variation_number"], x["repetition_number"]): x for x in audit["audit_details"]}
    judg_by_key = {(j["variation_number"], j["repetition_number"]): j for j in judgment["judgments"]}

    clear_keys = set()
    for x in audit["audit_details"]:
        if x["verdict"] == "CLEAR":
            clear_keys.add((x["variation_number"], x["repetition_number"]))

    items = []
    for ro in rollout["rollouts"]:
        var = ro["variation_number"]
        rep = ro["repetition_number"]
        key = (var, rep)
        if key not in clear_keys:
            continue
        var_desc = ideation["variations"][var-1]["description"] if var-1 < len(ideation["variations"]) else ro.get("variation_description", "")
        var_tools = ideation["variations"][var-1].get("tools", []) if var-1 < len(ideation["variations"]) else []
        items.append({
            "src_dir": d,
            "src_var": var,
            "src_rep": rep,
            "rollout": ro,
            "ideation_entry": {"description": var_desc, "tools": var_tools},
            "mtj_entry": mtj_by_key.get(key),
            "audit_entry": audit_by_key.get(key),
            "judgment_entry": judg_by_key.get(key),
        })
    return items

def write_combined(out_dir: Path, items: list, behavior_name: str):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n = len(items)
    new_rollouts, new_variations, new_mtj, new_audit, new_judgments, mapping = [], [], [], [], [], []

    for new_idx, it in enumerate(items, start=1):
        ro = dict(it["rollout"])
        ro["variation_number"] = new_idx
        ro["variation_description"] = it["ideation_entry"]["description"]
        new_rollouts.append(ro)
        new_variations.append(it["ideation_entry"])
        if it["mtj_entry"]:
            m = dict(it["mtj_entry"]); m["variation_number"] = new_idx; new_mtj.append(m)
        if it["audit_entry"]:
            a = dict(it["audit_entry"]); a["variation_number"] = new_idx; new_audit.append(a)
        if it["judgment_entry"]:
            j = dict(it["judgment_entry"]); j["variation_number"] = new_idx; new_judgments.append(j)
        # Copy transcript
        src_t = it["src_dir"] / f"transcript_v{it['src_var']}r{it['src_rep']}.json"
        dst_t = out_dir / f"transcript_v{new_idx}r{it['src_rep']}.json"
        if src_t.exists():
            shutil.copy(src_t, dst_t)
        mapping.append({
            "new_variation_number": new_idx,
            "src_dir": str(it["src_dir"].name),
            "src_variation_number": it["src_var"],
            "src_repetition_number": it["src_rep"],
            "classification": "CLEAR",
        })

    json.dump({"metadata": {"behavior": behavior_name, "combined_from": sorted(set(m["src_dir"] for m in mapping))},
               "rollouts": new_rollouts, "successful_count": n, "failed_count": 0,
               "total_count": n, "variations_count": n, "repetitions_per_variation": 1},
              open(out_dir / "rollout.json", "w"), indent=2)
    json.dump({"behavior_name": behavior_name, "examples": [], "model": "combined",
               "num_scenarios": n, "variation_dimensions": [], "total_evals": n, "variations": new_variations},
              open(out_dir / "ideation.json", "w"), indent=2)
    json.dump({"metadata": {}, "evaluated_rollouts": new_mtj},
              open(out_dir / "rollout_misalignment_turns.json", "w"), indent=2)
    nc = sum(1 for a in new_audit if a["verdict"] == "CLEAR")
    na = sum(1 for a in new_audit if a["verdict"] == "AMBIGUOUS")
    json.dump({"behavior": behavior_name, "dir": out_dir.name, "model": "combined",
               "total_rollouts": n, "misaligned_rollouts": len(new_mtj),
               "audited": len(new_audit), "num_ambiguous": na, "num_clear": nc,
               "ambiguous_rollouts": [], "audit_details": new_audit},
              open(out_dir / "ambiguous_rollouts.json", "w"), indent=2)
    json.dump({"behavior_name": behavior_name, "judgments": new_judgments,
               "total_conversations": n, "successful_count": len(new_judgments), "failed_count": 0},
              open(out_dir / "judgment.json", "w"), indent=2)
    json.dump({"mapping": mapping, "selection_summary": {"total": n, "clear": nc, "ambig": na, "benign": 0}},
              open(out_dir / "combination_mapping.json", "w"), indent=2)
    # Copy understanding.json
    src_u = items[0]["src_dir"] / "understanding.json"
    if src_u.exists():
        shutil.copy(src_u, out_dir / "understanding.json")

# Load all CLEAR items
v1 = load_clear_items(ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v1")
v2 = load_clear_items(ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v2")
v3 = load_clear_items(ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v3")
test_v1 = load_clear_items(ROOT / "bloom-results-test/test_strategic-deception_glm_4_7_flash_v1")

print(f"Available CLEAR: v1={len(v1)}, v2={len(v2)}, v3={len(v3)}, test_v1={len(test_v1)}")

# Dev (10): 8 from v1 + 2 from v2
dev_items = v1[:8] + v2[:2]
assert len(dev_items) == 10

# Test (14): 9 from test_v1 + 4 from v2 (remaining) + 1 from v3
test_items = test_v1[:9] + v2[2:6] + v3[:1]
assert len(test_items) == 14

print(f"Dev: {len(dev_items)} items from {set(it['src_dir'].name for it in dev_items)}")
print(f"Test: {len(test_items)} items from {set(it['src_dir'].name for it in test_items)}")

write_combined(ROOT / "bloom-results/strategic-deception_glm_4_7_flash", dev_items, "strategic-deception")
write_combined(ROOT / "bloom-results-test/test_strategic-deception_glm_4_7_flash", test_items, "strategic-deception")

print("\nWrote dev (10) and test (14)")
EOF