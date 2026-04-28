"""
Drift diagnostic: for every pure-wait pair (actions=[]), compute
per-vital before→after delta and aggregate by pathology × severity.

Run:
    python -m rule_engine._drift_diag [transitions_dir]
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def _iter_pairs(root: str):
    for fp in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        if fp.endswith("schema.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            case = json.load(f)
        cat = os.path.basename(os.path.dirname(fp))
        for pair in case["pairs"]:
            yield cat, case["case_id"], pair


VITALS = ("HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T")


def _direction(delta: float, dz: float) -> str:
    if abs(delta) <= dz:
        return "stable"
    return "up" if delta > 0 else "down"


DIR_DZ = {"HR": 3, "BP_sys": 5, "BP_dia": 5, "RR": 1, "O2Sat": 1, "T": 0.1}


def run(root: str) -> None:
    pure_wait_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pure_wait_by_path: dict[str, list[dict]] = defaultdict(list)

    total = 0
    pure_wait = 0
    for cat, case_id, pair in _iter_pairs(root):
        total += 1
        if pair["actions"]:
            continue
        if pair["duration_s"] <= 0:
            continue
        pure_wait += 1
        b = pair["before"]
        a = pair["after"]
        path = b["mechanism"]["pathology"]
        name, sev = path["name"], path["severity"]
        row = {
            "cat": cat,
            "case_id": case_id,
            "pair_id": pair["id"],
            "dur": pair["duration_s"],
            "before_HR": b["vitals"]["HR"],
            "after_HR": a["vitals"]["HR"],
        }
        for v in VITALS:
            row[f"d{v}"] = a["vitals"][v] - b["vitals"][v]
        pure_wait_by_key[(name, sev)].append(row)
        pure_wait_by_path[name].append(row)

    print("=" * 78)
    print(f"Pure-wait pair diagnostic — {pure_wait}/{total} pairs")
    print("=" * 78)

    # Pathology × severity table
    print()
    print(f"{'pathology':<32} {'sev':<8} {'n':>3}  "
          f"{'dHR':>7} {'dBPs':>7} {'dBPd':>7} {'dRR':>6} {'dO2':>6} {'dT':>6}")
    print("-" * 78)

    # Sort by pathology, grouped
    keys = sorted(pure_wait_by_key.keys())
    for k in keys:
        rows = pure_wait_by_key[k]
        name, sev = k
        n = len(rows)
        if n < 2:
            continue
        mean = {v: statistics.mean(r[f"d{v}"] for r in rows) for v in VITALS}
        print(f"{name:<32} {sev:<8} {n:>3}  "
              f"{mean['HR']:>+7.1f} {mean['BP_sys']:>+7.1f} {mean['BP_dia']:>+7.1f} "
              f"{mean['RR']:>+6.1f} {mean['O2Sat']:>+6.1f} {mean['T']:>+6.2f}")

    # Single-row pathologies summarized separately
    print()
    print(f"Pathologies with only 1 pure-wait pair (per severity) — showing totals:")
    for k in keys:
        rows = pure_wait_by_key[k]
        if len(rows) == 1:
            r = rows[0]
            name, sev = k
            print(f"  {name}/{sev}: {r['cat']}/{r['case_id']}/{r['pair_id']} "
                  f"dur={r['dur']} dHR={r['dHR']:+.0f} dBPs={r['dBP_sys']:+.0f} "
                  f"dO2={r['dO2Sat']:+.0f} dRR={r['dRR']:+.0f}")

    # Overall totals by pathology (all severities)
    print()
    print("=" * 78)
    print("Pathology totals (all severities combined)")
    print("=" * 78)
    total_rows = [(n, len(pure_wait_by_path[n])) for n in pure_wait_by_path]
    total_rows.sort(key=lambda x: -x[1])
    print(f"{'pathology':<32} {'n':>3}  "
          f"{'dHR':>7} {'dBPs':>7} {'dBPd':>7} {'dRR':>6} {'dO2':>6} {'dT':>6}")
    print("-" * 78)
    for name, n in total_rows:
        rows = pure_wait_by_path[name]
        mean = {v: statistics.mean(r[f"d{v}"] for r in rows) for v in VITALS}
        print(f"{name:<32} {n:>3}  "
              f"{mean['HR']:>+7.1f} {mean['BP_sys']:>+7.1f} {mean['BP_dia']:>+7.1f} "
              f"{mean['RR']:>+6.1f} {mean['O2Sat']:>+6.1f} {mean['T']:>+6.2f}")

    # Arrest emergence pairs (HR>0 → HR=0)
    print()
    print("=" * 78)
    print("Pure-wait arrest emergence (before.HR>0 -> after.HR=0)")
    print("=" * 78)
    for name, rows in pure_wait_by_path.items():
        for r in rows:
            if r["before_HR"] > 0 and r["after_HR"] == 0:
                print(f"  {r['cat']}/{r['case_id']}/{r['pair_id']} path={name} "
                      f"HR {r['before_HR']:.0f}→0 dBPs={r['dBP_sys']:+.0f} "
                      f"dO2={r['dO2Sat']:+.0f} dur={r['dur']}")

    # Arrest resolution pairs (HR=0 → HR>0)
    print()
    print("=" * 78)
    print("Pure-wait arrest resolution (before.HR=0 -> after.HR>0)")
    print("=" * 78)
    for name, rows in pure_wait_by_path.items():
        for r in rows:
            if r["before_HR"] == 0 and r["after_HR"] > 0:
                print(f"  {r['cat']}/{r['case_id']}/{r['pair_id']} path={name} "
                      f"HR 0→{r['after_HR']:.0f} dBPs={r['dBP_sys']:+.0f} "
                      f"dO2={r['dO2Sat']:+.0f} dur={r['dur']}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "transitions"
    run(root)
