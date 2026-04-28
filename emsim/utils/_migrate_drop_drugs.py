"""One-shot migration: strip `state.drugs[]` fields from every transition JSON.

After this runs, drugs appear only in `pair.actions[]` (as bolus events) and
the engine has no corpus-level drug memory across pair calls.

Safe to re-run (idempotent): deleting a missing key is a no-op.

Usage:
    python _migrate_drop_drugs.py
"""
from __future__ import annotations

import json
import os

TRANSITIONS = "transitions"


def main() -> None:
    n_cases = 0
    n_pairs = 0
    n_drug_entries_removed = 0
    n_initial_drug_entries_removed = 0

    for cat in sorted(os.listdir(TRANSITIONS)):
        cat_path = os.path.join(TRANSITIONS, cat)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".json") or fname == "schema.json":
                continue
            path = os.path.join(cat_path, fname)
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)

            init = case.get("initial_state") or {}
            if "drugs" in init:
                n_initial_drug_entries_removed += len(init["drugs"] or [])
                del init["drugs"]

            for pair in case.get("pairs", []):
                for side in ("before", "after"):
                    st = pair.get(side) or {}
                    if "drugs" in st:
                        n_drug_entries_removed += len(st["drugs"] or [])
                        del st["drugs"]
                n_pairs += 1

            with open(path, "w", encoding="utf-8") as f:
                json.dump(case, f, indent=2, ensure_ascii=False)
                f.write("\n")
            n_cases += 1

    print(f"Cases migrated       : {n_cases}")
    print(f"Pairs touched        : {n_pairs}")
    print(f"Drug entries removed :")
    print(f"  from initial_state : {n_initial_drug_entries_removed}")
    print(f"  from before/after  : {n_drug_entries_removed}")
    print(f"  total              : {n_initial_drug_entries_removed + n_drug_entries_removed}")


if __name__ == "__main__":
    main()
