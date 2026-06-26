"""Combine strategic-deception runs into dev/test sets with selection priority CLEAR > benign > AMBIGUOUS.

Dev (20):  former dev (v1) + new v2  → bloom-results/strategic-deception_glm_4_7_flash
Test (40): new v3 + former test (v1) → bloom-results-test/test_strategic-deception_glm_4_7_flash

Selection priority: CLEAR misaligned, then benign (unflagged by turn judge), then AMBIGUOUS.
Variation indices reassigned 1..N. All cross-references in mtj/audit/judgment updated.
"""
import json, shutil, random
from pathlib import Path

random.seed(42)

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/bloom")

def load_dir(d: Path):
    """Return a per-rollout dict with all metadata needed for selection + emit."""
    rollout = json.load(open(d / "rollout.json"))
    ideation = json.load(open(d / "ideation.json"))
    mtj = json.load(open(d / "rollout_misalignment_turns.json"))
    audit = json.load(open(d / "ambiguous_rollouts.json"))
    judgment = json.load(open(d / "judgment.json"))

    # Index by (variation_number, repetition_number)
    mtj_by_key = {(r["variation_number"], r["repetition_number"]): r for r in mtj["evaluated_rollouts"]}
    audit_by_key = {(x["variation_number"], x["repetition_number"]): x for x in audit["audit_details"]}
    judgment_by_key = {(j["variation_number"], j["repetition_number"]): j for j in judgment["judgments"]}

    items = []
    for ro in rollout["rollouts"]:
        var = ro["variation_number"]
        rep = ro["repetition_number"]
        key = (var, rep)
        mtj_entry = mtj_by_key.get(key)
        audit_entry = audit_by_key.get(key)
        judg_entry = judgment_by_key.get(key)
        # Classification:
        #   CLEAR    = audit verdict CLEAR
        #   AMBIG    = audit verdict AMBIGUOUS
        #   BENIGN   = no misaligned turns flagged at all (not in audit)
        if audit_entry is None or not (mtj_entry and mtj_entry.get("misaligned_turns")):
            cls = "BENIGN"
        elif audit_entry["verdict"] == "CLEAR":
            cls = "CLEAR"
        elif audit_entry["verdict"] == "AMBIGUOUS":
            cls = "AMBIG"
        else:
            cls = "BENIGN"
        # Variation description from ideation
        var_desc = ideation["variations"][var-1]["description"] if var-1 < len(ideation["variations"]) else ro.get("variation_description", "")
        var_tools = ideation["variations"][var-1].get("tools", []) if var-1 < len(ideation["variations"]) else []
        items.append({
            "src_dir": d,
            "src_var": var,
            "src_rep": rep,
            "cls": cls,
            "rollout": ro,
            "ideation_entry": {"description": var_desc, "tools": var_tools},
            "mtj_entry": mtj_entry,
            "audit_entry": audit_entry,
            "judgment_entry": judg_entry,
        })
    return items, ideation, mtj["metadata"], judgment

def select_items(pools_with_priority: list[tuple[str, list[dict]]], n_total: int):
    """pools_with_priority: list of (label, items) where label hints provenance for benign/ambig preference order.
    Returns selected list in canonical order: all CLEAR first, then benign (priority pools first), then AMBIG.
    """
    all_items = []
    for label, items in pools_with_priority:
        for it in items:
            it["src_label"] = label
            all_items.append(it)
    clear = [it for it in all_items if it["cls"] == "CLEAR"]
    benign = [it for it in all_items if it["cls"] == "BENIGN"]
    ambig = [it for it in all_items if it["cls"] == "AMBIG"]

    # Priority within benign/ambig: order pools from list (first pool = preferred), shuffle within pool for fairness.
    pool_order = [label for label, _ in pools_with_priority]
    def by_pool(items):
        out = []
        for label in pool_order:
            sub = [it for it in items if it["src_label"] == label]
            random.shuffle(sub)
            out.extend(sub)
        return out
    benign_ordered = by_pool(benign)
    ambig_ordered = by_pool(ambig)

    chosen = list(clear)
    need = n_total - len(chosen)
    chosen += benign_ordered[:need]
    need = n_total - len(chosen)
    chosen += ambig_ordered[:need]
    assert len(chosen) == n_total, f"got {len(chosen)} want {n_total}"
    return chosen, len(clear), min(len(benign_ordered), max(0, n_total - len(clear))), max(0, n_total - len(clear) - len(benign_ordered))

def write_combined(out_dir: Path, chosen: list[dict], src_judgment_template: dict, src_mtj_meta: dict, behavior_name: str):
    """Write combined dir with all 5 main files + transcript files, reindexed 1..N."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n = len(chosen)
    new_rollouts, new_variations, new_mtj, new_audit_details, new_judgments = [], [], [], [], []
    mapping = []  # for traceability

    audited_count = 0
    num_clear = 0
    num_ambig = 0

    for new_idx, it in enumerate(chosen, start=1):
        ro = dict(it["rollout"])
        ro["variation_number"] = new_idx
        ro["variation_description"] = it["ideation_entry"]["description"]
        new_rollouts.append(ro)
        new_variations.append(it["ideation_entry"])
        # MTJ: only include if entry exists; reindex
        if it["mtj_entry"] is not None:
            m = dict(it["mtj_entry"])
            m["variation_number"] = new_idx
            new_mtj.append(m)
        # Audit: only include if entry exists; reindex
        if it["audit_entry"] is not None:
            a = dict(it["audit_entry"])
            a["variation_number"] = new_idx
            new_audit_details.append(a)
            audited_count += 1
            if a["verdict"] == "CLEAR":
                num_clear += 1
            elif a["verdict"] == "AMBIGUOUS":
                num_ambig += 1
        # Judgment: include if entry exists
        if it["judgment_entry"] is not None:
            j = dict(it["judgment_entry"])
            j["variation_number"] = new_idx
            new_judgments.append(j)
        # Copy transcript file if present
        src_transcript = it["src_dir"] / f"transcript_v{it['src_var']}r{it['src_rep']}.json"
        dst_transcript = out_dir / f"transcript_v{new_idx}r{it['src_rep']}.json"
        if src_transcript.exists():
            shutil.copy(src_transcript, dst_transcript)
        mapping.append({
            "new_variation_number": new_idx,
            "src_dir": str(it["src_dir"].name),
            "src_variation_number": it["src_var"],
            "src_repetition_number": it["src_rep"],
            "classification": it["cls"],
        })

    # rollout.json
    new_rollout_obj = {
        "metadata": {"behavior": behavior_name, "combined_from": list(set(m["src_dir"] for m in mapping))},
        "rollouts": new_rollouts,
        "successful_count": n,
        "failed_count": 0,
        "total_count": n,
        "variations_count": n,
        "repetitions_per_variation": 1,
    }
    json.dump(new_rollout_obj, open(out_dir / "rollout.json", "w"), indent=2)

    # ideation.json
    new_ideation = {
        "behavior_name": behavior_name,
        "examples": [],
        "model": "combined",
        "num_scenarios": n,
        "variation_dimensions": [],
        "total_evals": n,
        "variations": new_variations,
    }
    json.dump(new_ideation, open(out_dir / "ideation.json", "w"), indent=2)

    # rollout_misalignment_turns.json
    new_mtj_obj = {
        "metadata": dict(src_mtj_meta),
        "evaluated_rollouts": new_mtj,
    }
    json.dump(new_mtj_obj, open(out_dir / "rollout_misalignment_turns.json", "w"), indent=2)

    # ambiguous_rollouts.json
    new_audit = {
        "behavior": behavior_name,
        "dir": out_dir.name,
        "model": "combined",
        "total_rollouts": n,
        "misaligned_rollouts": sum(1 for it in chosen if it["mtj_entry"] and it["mtj_entry"].get("misaligned_turns")),
        "audited": audited_count,
        "num_ambiguous": num_ambig,
        "num_clear": num_clear,
        "ambiguous_rollouts": [a for a in new_audit_details if a["verdict"] == "AMBIGUOUS"],
        "audit_details": new_audit_details,
    }
    json.dump(new_audit, open(out_dir / "ambiguous_rollouts.json", "w"), indent=2)

    # judgment.json
    new_judg = dict(src_judgment_template)
    new_judg["judgments"] = new_judgments
    new_judg["total_conversations"] = n
    new_judg["successful_count"] = len(new_judgments)
    new_judg["failed_count"] = n - len(new_judgments)
    json.dump(new_judg, open(out_dir / "judgment.json", "w"), indent=2)

    # mapping.json (traceability)
    json.dump({"mapping": mapping, "selection_summary": {"total": n, "clear": num_clear, "ambig": num_ambig, "benign": n - audited_count}},
              open(out_dir / "combination_mapping.json", "w"), indent=2)

    # Copy understanding.json from one source dir (not reindexable)
    src_understanding = chosen[0]["src_dir"] / "understanding.json"
    if src_understanding.exists():
        shutil.copy(src_understanding, out_dir / "understanding.json")


# ======================================================================
# DEV: former dev (v1) + new v2 → strategic-deception_glm_4_7_flash
# ======================================================================
print("=== DEV ===")
v1_dir = ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v1"
v2_dir = ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v2"
v1_items, _, v1_mtj_meta, v1_judg = load_dir(v1_dir)
v2_items, _, v2_mtj_meta, v2_judg = load_dir(v2_dir)
print(f"v1: {len(v1_items)} (CLEAR={sum(1 for x in v1_items if x['cls']=='CLEAR')}, BENIGN={sum(1 for x in v1_items if x['cls']=='BENIGN')}, AMBIG={sum(1 for x in v1_items if x['cls']=='AMBIG')})")
print(f"v2: {len(v2_items)} (CLEAR={sum(1 for x in v2_items if x['cls']=='CLEAR')}, BENIGN={sum(1 for x in v2_items if x['cls']=='BENIGN')}, AMBIG={sum(1 for x in v2_items if x['cls']=='AMBIG')})")
# Prefer v2 for benign/ambig (newer, more aligned with target prompt regime)
chosen_dev, nc, nb, na = select_items([("v2", v2_items), ("v1", v1_items)], 20)
print(f"Dev selection: CLEAR={nc} BENIGN={nb} AMBIG={na}")
out_dev = ROOT / "bloom-results/strategic-deception_glm_4_7_flash"
write_combined(out_dev, chosen_dev, v2_judg, v2_mtj_meta, "strategic-deception")
print(f"Wrote {out_dev}")

# ======================================================================
# TEST: new v3 + former test (v1) → test_strategic-deception_glm_4_7_flash
# ======================================================================
print("\n=== TEST ===")
v3_dir = ROOT / "bloom-results/strategic-deception_glm_4_7_flash_v3"
test_v1_dir = ROOT / "bloom-results-test/test_strategic-deception_glm_4_7_flash_v1"
v3_items, _, v3_mtj_meta, v3_judg = load_dir(v3_dir)
test_items, _, test_mtj_meta, test_judg = load_dir(test_v1_dir)
print(f"v3: {len(v3_items)} (CLEAR={sum(1 for x in v3_items if x['cls']=='CLEAR')}, BENIGN={sum(1 for x in v3_items if x['cls']=='BENIGN')}, AMBIG={sum(1 for x in v3_items if x['cls']=='AMBIG')})")
print(f"test_v1: {len(test_items)} (CLEAR={sum(1 for x in test_items if x['cls']=='CLEAR')}, BENIGN={sum(1 for x in test_items if x['cls']=='BENIGN')}, AMBIG={sum(1 for x in test_items if x['cls']=='AMBIG')})")
# Prefer v3 for benign/ambig (newer)
chosen_test, nc, nb, na = select_items([("v3", v3_items), ("test_v1", test_items)], 40)
print(f"Test selection: CLEAR={nc} BENIGN={nb} AMBIG={na}")
out_test = ROOT / "bloom-results-test/test_strategic-deception_glm_4_7_flash"
write_combined(out_test, chosen_test, v3_judg, v3_mtj_meta, "strategic-deception")
print(f"Wrote {out_test}")
