"""
Phase 4 drift calibration test.

For every pure-wait pair (actions=[], duration_s>0) run the full engine
(decode → drift → encode) and compare predicted `after.vitals` against
ground truth. Report per-pathology and per-vital MAE so coefficients can
be tuned.

Run:
    python -m rule_engine._drift_test [transitions_dir]
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rule_engine.engine import RuleEngine
    from emsim_eval import VITAL_TOLERANCE, VITAL_DIR_DEADZONE
else:
    from .engine import RuleEngine
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from emsim_eval import VITAL_TOLERANCE, VITAL_DIR_DEADZONE


VITALS = list(VITAL_TOLERANCE.keys())


def _direction(delta: float, dz: float) -> str:
    if abs(delta) <= dz:
        return "stable"
    return "up" if delta > 0 else "down"


def _iter_pairs(root: str):
    for fp in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        if fp.endswith("schema.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            case = json.load(f)
        cat = os.path.basename(os.path.dirname(fp))
        for pair in case["pairs"]:
            yield cat, case["case_id"], pair


def run(root: str) -> None:
    engine = RuleEngine()

    per_pair: list[dict] = []
    per_vital_err: dict[str, list[float]] = {v: [] for v in VITALS}
    by_path: dict[str, list[dict]] = defaultdict(list)

    for cat, case_id, pair in _iter_pairs(root):
        if pair["actions"]:
            continue
        if pair["duration_s"] <= 0:
            continue
        try:
            pred = engine.step(pair["before"], [], pair["duration_s"])
        except Exception as e:
            print(f"ERROR {cat}/{case_id}/{pair['id']}: {e}")
            continue
        row = {
            "cat": cat,
            "case_id": case_id,
            "pair_id": pair["id"],
            "path": pair["before"]["mechanism"]["pathology"]["name"],
            "sev": pair["before"]["mechanism"]["pathology"]["severity"],
            "dur": pair["duration_s"],
        }
        all_in = True
        all_dir = True
        for v in VITALS:
            err = abs(pred["vitals"][v] - pair["after"]["vitals"][v])
            row[f"{v}_err"] = err
            row[f"{v}_pred"] = pred["vitals"][v]
            row[f"{v}_actual"] = pair["after"]["vitals"][v]
            row[f"{v}_before"] = pair["before"]["vitals"][v]
            per_vital_err[v].append(err)
            if err > VITAL_TOLERANCE[v]:
                all_in = False
            dz = VITAL_DIR_DEADZONE[v]
            da = _direction(pair["after"]["vitals"][v] - pair["before"]["vitals"][v], dz)
            dp = _direction(pred["vitals"][v] - pair["before"]["vitals"][v], dz)
            if da != dp:
                all_dir = False
            row[f"{v}_dirmatch"] = da == dp
        row["vitals_in_tol"] = all_in
        row["dirs_match"] = all_dir
        row["strict"] = all_in and all_dir
        per_pair.append(row)
        by_path[row["path"]].append(row)

    n = len(per_pair)
    n_tol = sum(1 for r in per_pair if r["vitals_in_tol"])
    n_dir = sum(1 for r in per_pair if r["dirs_match"])
    n_strict = sum(1 for r in per_pair if r["strict"])

    print("=" * 78)
    print(f"Pure-wait drift test — {n} pairs")
    print("=" * 78)
    print(f"Vitals-all-in-tolerance : {n_tol}/{n}  ({100*n_tol/max(n,1):.1f}%)")
    print(f"Vitals-dirs-all-match   : {n_dir}/{n}  ({100*n_dir/max(n,1):.1f}%)")
    print(f"Both (strict-equivalent): {n_strict}/{n}  ({100*n_strict/max(n,1):.1f}%)")
    print()

    print("Per-vital MAE / P90 / >tol count:")
    print(f"  {'vital':<8} {'tol':>6} {'MAE':>7} {'P50':>7} {'P90':>7} {'>tol':>6}")
    for v in VITALS:
        errs = sorted(per_vital_err[v])
        mean = sum(errs) / max(len(errs), 1)
        p50 = errs[int(0.5 * len(errs))] if errs else 0
        p90 = errs[int(0.9 * len(errs))] if errs else 0
        over = sum(1 for e in errs if e > VITAL_TOLERANCE[v])
        print(f"  {v:<8} {VITAL_TOLERANCE[v]:>6} {mean:>7.2f} {p50:>7.2f} "
              f"{p90:>7.2f} {over:>6}")

    # Per-pathology
    print()
    print("Per-pathology (pathologies with >= 3 pairs):")
    print(f"  {'pathology':<32} {'n':>3} {'hit':>4}  "
          f"{'HR':>6} {'BPs':>6} {'BPd':>6} {'RR':>5} {'O2':>5} {'T':>5}")
    keys = sorted(by_path.keys(), key=lambda k: -len(by_path[k]))
    for path in keys:
        rows = by_path[path]
        if len(rows) < 3:
            continue
        n_p = len(rows)
        hit = sum(1 for r in rows if r["strict"])
        mae = {v: sum(r[f"{v}_err"] for r in rows) / n_p for v in VITALS}
        print(f"  {path:<32} {n_p:>3} {hit:>3}/{n_p:<1} "
              f"{mae['HR']:>6.1f} {mae['BP_sys']:>6.1f} {mae['BP_dia']:>6.1f} "
              f"{mae['RR']:>5.1f} {mae['O2Sat']:>5.1f} {mae['T']:>5.2f}")

    # Failure mode distribution: how many vitals out-of-tol per failing pair
    print()
    print("Failure-mode distribution (pairs failing on N vitals):")
    fail_count = defaultdict(int)
    for r in per_pair:
        n_out = sum(1 for v in VITALS if r[f"{v}_err"] > VITAL_TOLERANCE[v])
        fail_count[n_out] += 1
    for n_out in sorted(fail_count.keys()):
        print(f"  {n_out} vital(s) out of tol : {fail_count[n_out]} pairs")

    # Worst vital per failing pair (which vital pushes pairs out)
    print()
    print("Near-miss pairs (fail on exactly 1 vital) — by vital:")
    near_miss = defaultdict(int)
    for r in per_pair:
        outs = [v for v in VITALS if r[f"{v}_err"] > VITAL_TOLERANCE[v]]
        if len(outs) == 1:
            near_miss[outs[0]] += 1
    for v in VITALS:
        print(f"  {v:<8} {near_miss[v]} near-miss pairs could be recovered")

    # Worst 15 pairs
    print()
    print("Worst-15 pure-wait pairs (by tol-normalized total error):")
    ranked = sorted(
        per_pair,
        key=lambda r: -sum(r[f"{v}_err"] / VITAL_TOLERANCE[v] for v in VITALS),
    )
    for r in ranked[:15]:
        worst = sorted(
            VITALS,
            key=lambda v: -r[f"{v}_err"] / VITAL_TOLERANCE[v],
        )[:3]
        worst_str = ", ".join(
            f"{v}:{r[f'{v}_before']:.0f}->{r[f'{v}_pred']:.0f}(act {r[f'{v}_actual']:.0f})"
            for v in worst
        )
        tag = f"{r['case_id']}/{r['pair_id']}"
        print(f"  {r['path']:<28}/{r['sev']:<8} {tag:<40} {worst_str}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "transitions"
    run(root)
