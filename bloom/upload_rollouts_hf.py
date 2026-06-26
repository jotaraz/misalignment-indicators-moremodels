"""Upload cross-lingual rollout variants to the HF dataset
`kzhou35/misalignment-indicators-bloom-rollouts`.

The English originals already live there as `bloom-results/` (val) and
`bloom-results-test/` (test). This pushes the language variants under the same
convention, e.g. `bloom-results-vi/`, `bloom-results-test-vi/`, `ood-xlingual/`.

Only the Vietnamese (vi) variants are uploaded by default — that is the language
reported in the paper. Pass --lang cn to also push the Chinese variants.

Auth: reads `HF token: hf_...` from the repo .env (or HF_TOKEN env var).

Usage:
  python -m bloom.upload_rollouts_hf            # vi: dev + test + ood
  python -m bloom.upload_rollouts_hf --dry-run
  python -m bloom.upload_rollouts_hf --only ood
"""
from __future__ import annotations

import argparse, os, re
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/workspace-vast/kaiwen/misalignment-indicators")
BLOOM = ROOT / "bloom"
REPO_ID = "kzhou35/misalignment-indicators-bloom-rollouts"

# (local dir, path_in_repo, allow_patterns) per split, keyed by language.
def plan(lang):
    return {
        "dev":  (BLOOM / f"bloom-results-{lang}",      f"bloom-results-{lang}",      None),
        "test": (BLOOM / f"bloom-results-test-{lang}", f"bloom-results-test-{lang}", None),
        # ood-xlingual holds EN + cn + vi side by side; upload only this lang's dirs,
        # keeping them under ood-xlingual/ to match the local layout.
        "ood":  (BLOOM / "ood-xlingual",               "ood-xlingual",              [f"*_{lang}/*"]),
    }


def read_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = (ROOT / ".env").read_text()
    m = re.search(r"hf_[A-Za-z0-9]+", env)
    if not m:
        raise SystemExit("no HF token found (set HF_TOKEN or add 'hf_...' to .env)")
    return m.group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "cn"])
    ap.add_argument("--only", choices=["dev", "test", "ood"], help="upload just one split")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=read_token())
    splits = plan(a.lang)
    keys = [a.only] if a.only else ["dev", "test", "ood"]

    for k in keys:
        local, repo_path, allow = splits[k]
        if not local.exists():
            print(f"  skip {k}: {local} missing"); continue
        n = sum(1 for _ in local.rglob("rollout.json"))
        print(f"[{k}] {local}  ->  {REPO_ID}:{repo_path}  ({n} rollout dirs"
              f"{', allow=' + str(allow) if allow else ''})")
        if a.dry_run:
            continue
        api.upload_folder(
            repo_id=REPO_ID, repo_type="dataset", folder_path=str(local),
            path_in_repo=repo_path, allow_patterns=allow,
            commit_message=f"Add {a.lang.upper()} {k} rollout variants ({repo_path})",
        )
        print(f"  done: {repo_path}")


if __name__ == "__main__":
    main()
