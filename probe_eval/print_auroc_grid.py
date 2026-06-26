"""Print AUROC grid: probes (rows) × eval datasets (columns)."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

PROBES = [
    "repe_glm_lr__repe_honesty__you_are_fact_sys__glm-9b-flash__20260215_190119",
    "rp_glm_lr__roleplaying__plain__glm-9b-flash__20260216_055449",
    "rp_glm_offpolicy_lr__roleplaying__offpolicy_train__glm-9b-flash__20260216_043957",
]

PROBE_SHORT = [
    "repe_honesty",
    "rp_onpolicy",
    "rp_offpolicy",
]

DATASETS = [
    "insider_trading__onpolicy__glm-9b-flash_bloom",
    "roleplaying__plain__glm-9b-flash_bloom",
    "sandbagging",
    "undermining_oversight",
]

DATASET_SHORT = [
    "insider_trad",
    "roleplaying",
    "sandbagging",
    "undermine_ov",
]


def load_auroc(probe: str, dataset: str) -> str:
    path = RESULTS_DIR / probe / dataset / "results.json"
    if not path.exists():
        return "  ---  "
    with open(path) as f:
        data = json.load(f)
    auroc = data.get("metrics", {}).get("auroc")
    if auroc is None or auroc != auroc:  # NaN check
        return "  NaN  "
    return f" {auroc:.3f} "


def main():
    col_w = 14
    probe_w = 16

    # Header
    header = f"{'probe':<{probe_w}}" + "".join(f"{d:^{col_w}}" for d in DATASET_SHORT)
    print(header)
    print("-" * len(header))

    # Rows
    for probe, short in zip(PROBES, PROBE_SHORT):
        row = f"{short:<{probe_w}}"
        for dataset in DATASETS:
            val = load_auroc(probe, dataset)
            row += f"{val:^{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
