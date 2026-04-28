"""
Phase 5 intervention diagnostic.

For every pair in `transitions/`:
  - run the engine
  - score against ground truth (strict, per-vital MAE, dir-match)
  - classify pair into slices: pure-wait / action-defib / action-non-defib
  - for each intervention name, collect pairs where that name is the only
    intervention in `actions[]` (drugs may still coexist) and report per-
    intervention strict + per-vital MAE

Also prints worst-20 per slice so the calibration loop has a focus list.

Run:
    python -m rule_engine._intervention_test [transitions_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rule_engine.engine import RuleEngine
    from emsim_eval import VITAL_TOLERANCE, VITAL_DIR_DEADZONE, evaluate_pair
else:
    from .engine import RuleEngine
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from emsim_eval import VITAL_TOLERANCE, VITAL_DIR_DEADZONE, evaluate_pair


VITALS = list(VITAL_TOLERANCE.keys())


def _iter_pairs(root: str):
    for fp in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        if fp.endswith("schema.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            case = json.load(f)
        cat = os.path.basename(os.path.dirname(fp))
        for pair in case["pairs"]:
            yield cat, case["case_id"], pair


def _classify(pair):
    """Return set of slice labels the pair belongs to."""
    labels = set()
    actions = pair.get("actions", [])
    intervs = [a for a in actions if a.get("type") == "intervention"]
    drugs = [a for a in actions if a.get("type") == "drug"]
    names = [a.get("name", "") for a in intervs]

    if not actions:
        labels.add("pure_wait")
    else:
        labels.add("action_any")
        if "defibrillate" in names:
            labels.add("action_defib")
        else:
            labels.add("action_non_defib")

    # "single-intervention" slice: exactly one intervention (drugs ok).
    if len(intervs) == 1:
        labels.add(f"only:{names[0]}")
    # and if no drugs either, "pure single intervention"
    if len(intervs) == 1 and not drugs:
        labels.add(f"solo:{names[0]}")

    return labels


def _score_slice(rows):
    if not rows:
        return None
    n = len(rows)
    strict = sum(1 for r in rows if r["pair_pass"])
    partial = sum(r["partial_score"] for r in rows) / n
    mae = {v: sum(r["vitals"][v]["abs_err"] for r in rows) / n for v in VITALS}
    dir_match = {v: sum(1 for r in rows if r["vitals"][v]["dir_match"]) / n
                 for v in VITALS}
    within = {v: sum(1 for r in rows if r["vitals"][v]["within_tol"]) / n
              for v in VITALS}
    return {
        "n": n, "strict": strict, "partial": partial,
        "mae": mae, "dir_match": dir_match, "within": within,
    }


def _fmt_row(label, s):
    return (f"  {label:<32} n={s['n']:>3}  strict={s['strict']:>3}/{s['n']:<3} "
            f"({100*s['strict']/s['n']:>5.1f}%)  "
            f"HR={s['mae']['HR']:>5.2f} BPs={s['mae']['BP_sys']:>5.2f} "
            f"BPd={s['mae']['BP_dia']:>5.2f} RR={s['mae']['RR']:>4.2f} "
            f"O2={s['mae']['O2Sat']:>5.2f} T={s['mae']['T']:>4.2f}")


def run(root: str) -> None:
    engine = RuleEngine()

    all_rows = []
    slice_rows: dict[str, list[dict]] = defaultdict(list)

    for cat, case_id, pair in _iter_pairs(root):
        try:
            pred = engine.step(pair["before"], pair["actions"], pair["duration_s"])
        except Exception as e:
            print(f"ERROR {cat}/{case_id}/{pair['id']}: {e}")
            continue
        r = evaluate_pair(pair, pred)
        r["category"] = cat
        r["case_id"] = case_id
        r["pair_id"] = pair["id"]
        r["n_actions"] = len(pair["actions"])
        r["duration_s"] = pair["duration_s"]
        r["action_names"] = [a.get("name", "") for a in pair["actions"]]
        r["_pair"] = pair
        all_rows.append(r)
        for label in _classify(pair):
            slice_rows[label].append(r)

    # --- Top-level slices ---
    print("=" * 100)
    print("Phase 5 intervention diagnostic")
    print("=" * 100)
    overall = _score_slice(all_rows)
    print(_fmt_row("OVERALL", overall))
    for name in ("pure_wait", "action_any", "action_defib", "action_non_defib"):
        s = _score_slice(slice_rows.get(name, []))
        if s:
            print(_fmt_row(name, s))

    # --- Per-intervention (both "only" = pair with exactly this one intervention;
    # drugs may coexist, and "solo" = no drugs either) ---
    print()
    print("Per-intervention slices:")
    print("  'only:X' = exactly one intervention (X), drugs may coexist")
    print("  'solo:X' = exactly one intervention (X), no drugs either")
    print()
    # sort by count desc
    inter_keys = sorted(
        (k for k in slice_rows if k.startswith("only:")),
        key=lambda k: -len(slice_rows[k]),
    )
    for k in inter_keys:
        s = _score_slice(slice_rows[k])
        print(_fmt_row(k, s))

    print()
    solo_keys = sorted(
        (k for k in slice_rows if k.startswith("solo:")),
        key=lambda k: -len(slice_rows[k]),
    )
    for k in solo_keys:
        s = _score_slice(slice_rows[k])
        print(_fmt_row(k, s))

    # --- Worst-20 action-non-defib ---
    print()
    print("Worst-20 action_non_defib pairs:")
    rows = slice_rows.get("action_non_defib", [])
    ranked = sorted(
        rows,
        key=lambda r: -sum(r["vitals"][v]["abs_err"] / VITAL_TOLERANCE[v] for v in VITALS),
    )
    for r in ranked[:20]:
        worst = sorted(
            VITALS,
            key=lambda v: -r["vitals"][v]["abs_err"] / VITAL_TOLERANCE[v],
        )[:3]
        worst_str = ", ".join(
            f"{v}:{r['vitals'][v]['before']:.0f}->"
            f"{r['vitals'][v]['predicted']:.0f}(act {r['vitals'][v]['actual']:.0f})"
            for v in worst
        )
        actions = ",".join(r["action_names"])
        tag = f"{r['case_id']}/{r['pair_id']}"
        print(f"  [{r['category']:<13}] {tag:<45} acts=[{actions:<40}] {worst_str}")

    # --- Worst-20 pure-wait for comparison ---
    print()
    print("Worst-20 pure-wait pairs:")
    rows = slice_rows.get("pure_wait", [])
    ranked = sorted(
        rows,
        key=lambda r: -sum(r["vitals"][v]["abs_err"] / VITAL_TOLERANCE[v] for v in VITALS),
    )
    for r in ranked[:20]:
        worst = sorted(
            VITALS,
            key=lambda v: -r["vitals"][v]["abs_err"] / VITAL_TOLERANCE[v],
        )[:3]
        worst_str = ", ".join(
            f"{v}:{r['vitals'][v]['before']:.0f}->"
            f"{r['vitals'][v]['predicted']:.0f}(act {r['vitals'][v]['actual']:.0f})"
            for v in worst
        )
        path = r["_pair"]["before"]["mechanism"]["pathology"]
        tag = f"{r['case_id']}/{r['pair_id']}"
        print(f"  [{r['category']:<13}] {tag:<45} path={path['name']:<20}/{path['severity']:<8} {worst_str}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    root = sys.argv[1] if len(sys.argv) > 1 else "transitions"
    run(root)
