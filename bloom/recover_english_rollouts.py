#!/usr/bin/env python3
"""Recover English rollout.json files from the per-rollout English transcript_v#r#.json
files, by swapping the (Chinese / translator-refusal) reasoning parts in
bloom-results-cn/<behavior>/rollout.json with the English reasoning from the
matching transcript files.

Output: bloom-results/<behavior>/rollout.json  (English reasoning restored).

Only the assistant `reasoning` content parts are replaced; every other field
(ids, timestamps, system/user/tool text, visible assistant text, top-level
metadata) is preserved verbatim from rollout.json. A rollout is only modified
when its transcript file is structurally identical (same events, roles, and
non-reasoning text); otherwise it is left as-is and reported.
"""
import json, re, glob, os, sys

CN_ROOT = os.path.join(os.path.dirname(__file__), "bloom-results-cn")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "bloom-results")
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def reasoning_parts_in_order(events):
    """Return list of (event_idx, part_idx) for every assistant reasoning part."""
    locs = []
    for ei, e in enumerate(events):
        m = e["edit"]["message"]
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for pi, p in enumerate(m["content"]):
                if p.get("type") == "reasoning":
                    locs.append((ei, pi))
    return locs


def structural_match(ev_ro, ev_tf):
    """True iff same #events, same roles, same content-part shape, and all
    non-reasoning text identical. Returns (ok, reason)."""
    if len(ev_ro) != len(ev_tf):
        return False, f"event count {len(ev_ro)} vs {len(ev_tf)}"
    for i, (a, b) in enumerate(zip(ev_ro, ev_tf)):
        ma, mb = a["edit"]["message"], b["edit"]["message"]
        if ma["role"] != mb["role"]:
            return False, f"role mismatch at event {i}: {ma['role']} vs {mb['role']}"
        ca, cb = ma["content"], mb["content"]
        if isinstance(ca, str) or isinstance(cb, str):
            if ca != cb:
                return False, f"str content differs at event {i} (role {ma['role']})"
            continue
        if len(ca) != len(cb):
            return False, f"content-part count differs at event {i}"
        for j, (pa, pb) in enumerate(zip(ca, cb)):
            if pa.get("type") != pb.get("type"):
                return False, f"part type differs at event {i} part {j}"
            if pa.get("type") != "reasoning":  # text / other: must be identical
                if pa.get("text") != pb.get("text"):
                    return False, f"non-reasoning text differs at event {i} part {j}"
    return True, ""


def count_cjk_reasoning(events):
    n = 0
    for e in events:
        m = e["edit"]["message"]
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for p in m["content"]:
                if p.get("type") == "reasoning" and CJK.search(p["reasoning"]):
                    n += 1
    return n


def process_behavior(behavior):
    cn_dir = os.path.join(CN_ROOT, behavior)
    roll_path = os.path.join(cn_dir, "rollout.json")
    tf_paths = sorted(glob.glob(os.path.join(cn_dir, "transcript_v*r*.json")))
    if not os.path.exists(roll_path) or not tf_paths:
        return None

    data = json.load(open(roll_path))
    by_vr = {}
    for r in data["rollouts"]:
        by_vr[(r["variation_number"], r["repetition_number"])] = r

    tf_by_vr = {}
    for p in tf_paths:
        m = re.search(r"transcript_v(\d+)r(\d+)\.json$", p)
        tf_by_vr[(int(m.group(1)), int(m.group(2)))] = p

    swapped = 0
    parts_swapped = 0
    failed = []          # (vr, reason)
    no_transcript = []   # vr present in rollout but no transcript file
    cjk_before = cjk_after = 0

    for vr, r in by_vr.items():
        ev_ro = r["transcript"]["events"]
        cjk_before += count_cjk_reasoning(ev_ro)
        if vr not in tf_by_vr:
            no_transcript.append(vr)
            cjk_after += count_cjk_reasoning(ev_ro)
            continue
        ev_tf = json.load(open(tf_by_vr[vr]))["events"]
        ok, reason = structural_match(ev_ro, ev_tf)
        if not ok:
            failed.append((vr, reason))
            cjk_after += count_cjk_reasoning(ev_ro)
            continue
        # Swap reasoning parts in place (positions are aligned by structural_match).
        locs = reasoning_parts_in_order(ev_ro)
        locs_tf = reasoning_parts_in_order(ev_tf)
        assert locs == locs_tf
        for (ei, pi) in locs:
            en = ev_tf[ei]["edit"]["message"]["content"][pi]["reasoning"]
            ev_ro[ei]["edit"]["message"]["content"][pi]["reasoning"] = en
            parts_swapped += 1
        swapped += 1
        cjk_after += count_cjk_reasoning(ev_ro)

    # Write recovered rollout.json
    out_dir = os.path.join(OUT_ROOT, behavior)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rollout.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    return {
        "behavior": behavior,
        "n_rollouts": len(by_vr),
        "n_transcripts": len(tf_by_vr),
        "swapped": swapped,
        "parts_swapped": parts_swapped,
        "failed": failed,
        "no_transcript": no_transcript,
        "cjk_before": cjk_before,
        "cjk_after": cjk_after,
        "out": os.path.join(out_dir, "rollout.json"),
    }


def main():
    behaviors = sys.argv[1:] or sorted(
        d for d in os.listdir(CN_ROOT) if os.path.isdir(os.path.join(CN_ROOT, d))
    )
    print(f"{'behavior':52s} roll  tf  swap  parts  cjk_before->after  fails  notf")
    grand = {"swapped": 0, "fail": 0, "notf": 0}
    for b in behaviors:
        res = process_behavior(b)
        if res is None:
            print(f"{b:52s} (no rollout.json or no transcript files — skipped)")
            continue
        print(f"{b:52s} {res['n_rollouts']:4d} {res['n_transcripts']:3d} "
              f"{res['swapped']:4d} {res['parts_swapped']:5d}  "
              f"{res['cjk_before']:5d} -> {res['cjk_after']:<5d}  "
              f"{len(res['failed']):4d}  {len(res['no_transcript']):4d}")
        grand["swapped"] += res["swapped"]
        grand["fail"] += len(res["failed"])
        grand["notf"] += len(res["no_transcript"])
        for vr, reason in res["failed"]:
            print(f"      ! FAIL v{vr[0]}r{vr[1]}: {reason}")
        if res["no_transcript"]:
            print(f"      . no transcript for: {sorted(res['no_transcript'])}")
    print(f"\nTOTAL: swapped={grand['swapped']} struct_fails={grand['fail']} "
          f"missing_transcript={grand['notf']}")


if __name__ == "__main__":
    main()
