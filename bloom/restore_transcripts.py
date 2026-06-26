"""Re-copy transcript files for dev/test combined dirs from original source dirs using mapping."""
import json, shutil
from pathlib import Path

ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/bloom")
TEST_ROOT = Path("/workspace-vast/kaiwen/persona_vectors/data_generation/bloom/bloom-results-test")

def src_path_for(src_dir_name: str) -> Path:
    if src_dir_name.startswith("test_"):
        return TEST_ROOT / src_dir_name
    return ROOT / "bloom-results" / src_dir_name

for label, p in [
    ("DEV",  ROOT / "bloom-results/strategic-deception_glm_4_7_flash"),
    ("TEST", TEST_ROOT / "test_strategic-deception_glm_4_7_flash"),
]:
    mapping = json.load(open(p / "combination_mapping.json"))
    # Delete any existing transcript files
    for f in p.glob("transcript_v*r1.json"):
        f.unlink()
    copied = 0
    missing = []
    for m in mapping["mapping"]:
        src_dir = src_path_for(m["src_dir"])
        src_var = m["src_variation_number"]
        rep = m["src_repetition_number"]
        new_var = m["new_variation_number"]
        src = src_dir / f"transcript_v{src_var}r{rep}.json"
        dst = p / f"transcript_v{new_var}r{rep}.json"
        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append((new_var, str(src)))
    print(f"{label}: copied {copied}, missing {len(missing)}")
    for nv, sp in missing[:5]:
        print(f"  miss new_var={nv} src={sp}")
