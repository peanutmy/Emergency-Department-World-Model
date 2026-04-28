"""
Round-trip accuracy test (Phase 3a).

For every pair's `before` state, decode → map and compare predicted vitals
against observed `before.vitals`. Target: on ≥ 95 % of pairs every vital
error is below `VITAL_TOLERANCE[v] / 2`.

Run from repo root:
    python -m rule_engine._roundtrip_test
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

# Allow running as a module or a script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rule_engine import io, mapping
    from rule_engine.hidden_state import ARREST_RHYTHMS
    from emsim_eval import VITAL_TOLERANCE
else:
    from . import io, mapping
    from .hidden_state import ARREST_RHYTHMS
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from emsim_eval import VITAL_TOLERANCE


VITALS = list(VITAL_TOLERANCE.keys())


def _predict_vitals(before: dict) -> dict:
    """decode_state → mapping → dict of 6 predicted vitals.

    Uses same call shape as Phase 3b `io.encode_state` for full-stack parity.
    """
    h, flags = io.decode_state(before)
    bv = before["vitals"]

    hr = mapping.HR_from_hidden(h, before_HR=bv["HR"])
    bp_sys, bp_dia = mapping.BP_from_hidden(h, flags)
    o2  = mapping.O2Sat_from_hidden(h, flags)
    if o2 is None:
        o2 = bv["O2Sat"]
    rr  = mapping.RR_from_hidden(h, flags)
    t = bv["T"]   # dt_s=0 for round-trip → T_from_hidden(h, T_before, 0) == T_before

    return {"HR": hr, "BP_sys": bp_sys, "BP_dia": bp_dia,
            "RR": rr, "O2Sat": o2, "T": t}, h.rhythm


def _iter_pairs(root: str):
    for fp in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        if fp.endswith("schema.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            case = json.load(f)
        cat = os.path.basename(os.path.dirname(fp))
        for pair in case["pairs"]:
            yield cat, case["case_id"], pair


def run_roundtrip(transitions_dir: str = "transitions") -> dict:
    per_vital_err: dict[str, list[float]] = {v: [] for v in VITALS}
    fail_pairs: list[dict] = []
    path_rhythm: dict[str, Counter] = defaultdict(Counter)  # {pathology: Counter(rhythm)}
    # Per-rhythm / pathology error tracking
    by_rhythm_err: dict[str, list[float]] = defaultdict(list)

    total = 0
    any_fail = 0
    for cat, case_id, pair in _iter_pairs(transitions_dir):
        total += 1
        before = pair["before"]
        predicted, rhythm = _predict_vitals(before)
        obs = before["vitals"]

        pair_fail = False
        row = {"category": cat, "case_id": case_id, "pair_id": pair["id"],
               "rhythm": rhythm,
               "pathology": before["mechanism"]["pathology"]["name"],
               "severity": before["mechanism"]["pathology"]["severity"]}

        # Arrest-rhythm HR round-trip is trivially pass when before.HR==0
        # (predicted=0 for asystole/VF/PEA). For PEA with before.HR>0 we
        # count it anyway; the mapping outputs 0 which is the Phase 2/3a
        # contract; Phase 3b flips PEA to preserve-monitor-HR.

        for v in VITALS:
            err = abs(predicted[v] - obs[v])
            per_vital_err[v].append(err)
            half = VITAL_TOLERANCE[v] / 2
            row[f"{v}_err"] = err
            if err > half:
                pair_fail = True
                fail_pairs.append({**row, "vital": v, "err": err,
                                   "predicted": predicted[v], "observed": obs[v]})

        if pair_fail:
            any_fail += 1
        path_rhythm[before["mechanism"]["pathology"]["name"]][rhythm] += 1
        by_rhythm_err[rhythm].append(sum(row[f"{v}_err"] / VITAL_TOLERANCE[v] for v in VITALS))

    # --- Summary ---
    print("=" * 72)
    print(f"Round-trip test — {total} pairs")
    print("=" * 72)
    print(f"Pairs passing ALL 6 vitals within tolerance/2: "
          f"{total - any_fail}/{total}  ({100*(total-any_fail)/total:.1f}%)")
    print()
    print(f"Per-vital error distribution (tolerance/2 thresholds in parens):")
    print(f"  {'vital':<8}  {'tol/2':>6}  {'mean':>7}  {'P50':>7}  {'P90':>7}  "
          f"{'P99':>7}  {'>tol/2':>7}")
    for v in VITALS:
        errs = per_vital_err[v]
        errs_sorted = sorted(errs)
        n = len(errs)
        mean = sum(errs) / n
        p50  = errs_sorted[int(0.50 * n)]
        p90  = errs_sorted[int(0.90 * n)]
        p99  = errs_sorted[min(n-1, int(0.99 * n))]
        tol_half = VITAL_TOLERANCE[v] / 2
        over = sum(1 for e in errs if e > tol_half)
        print(f"  {v:<8}  {tol_half:>6.1f}  {mean:>7.2f}  {p50:>7.2f}  {p90:>7.2f}  "
              f"{p99:>7.2f}  {over:>4} ({100*over/n:.1f}%)")

    # --- Top failures by pathology ---
    by_pathology = Counter()
    for f in fail_pairs:
        by_pathology[f["pathology"]] += 1
    print()
    print("Top 10 pathologies by failure count (vital > tol/2):")
    for p, n in by_pathology.most_common(10):
        print(f"  {n:3}  {p}")

    # --- Sample failure rows (first 10 unique case/pair) ---
    seen = set()
    print()
    print("Sample failure pairs (first 10 unique):")
    for f in fail_pairs:
        key = (f["case_id"], f["pair_id"])
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 10:
            break
        print(f"  [{f['category']:<13}] {f['case_id']}/{f['pair_id']}  "
              f"rhy={f['rhythm']:<10} path={f['pathology']:<25} "
              f"vital={f['vital']} err={f['err']:.2f} (pred={f['predicted']:.1f} obs={f['observed']:.1f})")

    return {
        "total": total, "any_fail": any_fail,
        "per_vital_err": per_vital_err,
        "fail_pairs": fail_pairs,
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    root = sys.argv[1] if len(sys.argv) > 1 else "transitions"
    run_roundtrip(root)
