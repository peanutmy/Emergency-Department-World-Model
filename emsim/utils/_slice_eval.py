"""Compute slice breakdown (pure_wait / drug_only / interv_only / action_non_defib / defib / arrest)
over an eval CSV by re-running the transitions directory in parallel to pull action types.

Usage:
    python _slice_eval.py pre_drug_removal.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

TRANSITIONS = "transitions"


def load_pair_metadata():
    """Return dict pair_key → {'has_drug':bool, 'has_interv':bool, 'has_defib':bool, 'arrest':bool}."""
    meta = {}
    for cat in sorted(os.listdir(TRANSITIONS)):
        cat_path = os.path.join(TRANSITIONS, cat)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".json") or fname == "schema.json":
                continue
            with open(os.path.join(cat_path, fname), encoding="utf-8") as f:
                case = json.load(f)
            for pair in case["pairs"]:
                key = (cat, case["case_id"], pair["id"])
                has_drug = any(a.get("type") == "drug" for a in pair["actions"])
                has_interv = any(a.get("type") == "intervention" for a in pair["actions"])
                has_defib = any(a.get("type") == "intervention"
                                and a.get("name") == "defibrillate"
                                for a in pair["actions"])
                arrest = pair["before"]["vitals"]["HR"] == 0
                meta[key] = {
                    "has_drug": has_drug,
                    "has_interv": has_interv,
                    "has_defib": has_defib,
                    "arrest": arrest,
                }
    return meta


def main():
    csv_path = sys.argv[1]
    meta = load_pair_metadata()

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["category"], row["case_id"], row["pair_id"])
            m = meta.get(key)
            if m is None:
                continue
            row.update(m)
            rows.append(row)

    def slice_stats(filtered, label):
        n = len(filtered)
        if n == 0:
            print(f"  {label:<20}: (empty)")
            return
        n_pass = sum(1 for r in filtered if r["pair_pass"] == "True")
        partial = sum(float(r["partial_score"]) for r in filtered) / n
        print(f"  {label:<20}: {n_pass:>3}/{n:<3} = {100*n_pass/n:5.1f}%   partial={partial:.3f}")

    print(f"File: {csv_path}")
    print(f"Total pairs: {len(rows)}")
    n_pass = sum(1 for r in rows if r["pair_pass"] == "True")
    partial = sum(float(r["partial_score"]) for r in rows) / max(len(rows), 1)
    print(f"  {'overall':<20}: {n_pass:>3}/{len(rows)} = {100*n_pass/len(rows):5.1f}%   partial={partial:.3f}")

    pure_wait = [r for r in rows if int(r["n_actions"]) == 0]
    drug_only = [r for r in rows if r["has_drug"] and not r["has_interv"]]
    interv_only = [r for r in rows if r["has_interv"] and not r["has_drug"]]
    interv_drug_mixed = [r for r in rows if r["has_interv"] and r["has_drug"]]
    action_non_defib = [r for r in rows if int(r["n_actions"]) > 0 and not r["has_defib"]]
    defib = [r for r in rows if r["has_defib"]]
    arrest = [r for r in rows if r["arrest"]]

    slice_stats(pure_wait,       "pure_wait")
    slice_stats(drug_only,       "drug_only")
    slice_stats(interv_only,     "interv_only")
    slice_stats(interv_drug_mixed,"mixed_interv_drug")
    slice_stats(action_non_defib,"action_non_defib")
    slice_stats(defib,           "defib")
    slice_stats(arrest,          "arrest")

    # Per-vital MAE (over scored pairs)
    print("\nPer-vital MAE:")
    for v in ["HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"]:
        errs = [float(r[f"{v}_abs_err"]) for r in rows if r.get(f"{v}_abs_err")]
        mae = sum(errs) / max(len(errs), 1)
        print(f"  {v:<8}: MAE={mae:>7.3f}   N={len(errs)}")


if __name__ == "__main__":
    main()
