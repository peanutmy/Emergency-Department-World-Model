"""
Phase 6 drug diagnostic — per-drug + per-composition slice analysis.

For every pair in `transitions/`:
  - run the engine
  - score against ground truth (strict, per-vital MAE, direction)
  - classify pair into slices:
      pure_wait / interv_only / drug_only / defib / mixed
      per-drug (solo drug-only subset)
      per-class (all pairs whose single drug maps to a given class)

Also prints worst-20 per slice and highlights drugs that regressed vs a
Phase 5 baseline CSV if provided (`--baseline phase5_baseline.csv`).

Run:
    python -m rule_engine._drug_test [transitions_dir]
    python -m rule_engine._drug_test transitions --baseline phase5_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rule_engine.engine import RuleEngine
    from rule_engine.drug_lib import DRUG_TO_CLASS
    from emsim_eval import VITAL_TOLERANCE, evaluate_pair
else:
    from .engine import RuleEngine
    from .drug_lib import DRUG_TO_CLASS
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from emsim_eval import VITAL_TOLERANCE, evaluate_pair


VITALS = list(VITAL_TOLERANCE.keys())


def _iter_pairs(root):
    for fp in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        if fp.endswith("schema.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            case = json.load(f)
        cat = os.path.basename(os.path.dirname(fp))
        for pair in case["pairs"]:
            yield cat, case["case_id"], pair


def _classify(pair):
    labels = set()
    actions = pair.get("actions", [])
    drugs = [a for a in actions if a.get("type") == "drug"]
    intervs = [a for a in actions if a.get("type") == "intervention"]
    names = [a.get("name", "") for a in intervs]

    if not actions:
        labels.add("pure_wait")
    elif drugs and not intervs:
        labels.add("drug_only")
    elif intervs and not drugs:
        labels.add("interv_only")
        if "defibrillate" in names:
            labels.add("action_defib")
        else:
            labels.add("action_non_defib")
    else:
        labels.add("mixed")

    # Per-drug slice: drug-only + exactly one drug.
    if drugs and not intervs and len(drugs) == 1:
        name = drugs[0].get("name", "")
        labels.add(f"drug:{name}")
        cls = DRUG_TO_CLASS.get(name)
        if cls:
            labels.add(f"class:{cls}")

    return labels


def _score_slice(rows):
    if not rows:
        return None
    n = len(rows)
    strict = sum(1 for r in rows if r["pair_pass"])
    mae = {v: sum(r["vitals"][v]["abs_err"] for r in rows) / n for v in VITALS}
    return {"n": n, "strict": strict, "mae": mae}


def _fmt(label, s, baseline=None):
    if not s:
        return ""
    base_str = ""
    if baseline:
        d_strict = s["strict"] - baseline["strict"]
        d_mae_hr = s["mae"]["HR"] - baseline["mae"]["HR"]
        d_mae_bp = s["mae"]["BP_sys"] - baseline["mae"]["BP_sys"]
        base_str = f" Δstrict={d_strict:+d} ΔHR={d_mae_hr:+.2f} ΔBPs={d_mae_bp:+.2f}"
    return (f"  {label:<28} n={s['n']:>3} strict={s['strict']:>3}/{s['n']:<3}"
            f"({100*s['strict']/s['n']:>5.1f}%) "
            f"HR={s['mae']['HR']:>5.2f} BPs={s['mae']['BP_sys']:>5.2f} "
            f"BPd={s['mae']['BP_dia']:>5.2f} RR={s['mae']['RR']:>4.2f} "
            f"O2={s['mae']['O2Sat']:>5.2f} T={s['mae']['T']:>4.2f}{base_str}")


def _load_baseline(path):
    """Load a previously-exported phase5_baseline.csv and re-derive per-slice stats."""
    if not path or not os.path.exists(path):
        return {}
    rows = []
    with open(path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if rec.get("error"):
                continue
            d = {"pair_pass": rec.get("pair_pass") == "True",
                 "vitals": {}, "_pair_id": f"{rec['case_id']}/{rec['pair_id']}"}
            for v in VITALS:
                try:
                    d["vitals"][v] = {
                        "abs_err": float(rec[f"{v}_abs_err"] or 0.0),
                        "within_tol": rec[f"{v}_within_tol"] == "True",
                        "dir_match": rec[f"{v}_dir_match"] == "True",
                    }
                except (KeyError, ValueError):
                    d["vitals"][v] = {"abs_err": 0.0, "within_tol": False, "dir_match": False}
            rows.append(d)
    return rows


def run(root, baseline_path=None):
    engine = RuleEngine()
    all_rows = []
    slice_rows = defaultdict(list)
    # Capture per-pair for regression comparison keyed by (case_id, pair_id).
    by_pair_id = {}

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
        r["_pair_id"] = f"{case_id}/{pair['id']}"
        all_rows.append(r)
        by_pair_id[r["_pair_id"]] = r
        for label in _classify(pair):
            slice_rows[label].append(r)

    # --- Top-level slices ---
    print("=" * 110)
    print("Phase 6 drug diagnostic")
    print("=" * 110)
    overall = _score_slice(all_rows)
    print(_fmt("OVERALL", overall))
    for name in ("pure_wait", "interv_only", "drug_only", "mixed",
                 "action_defib", "action_non_defib"):
        s = _score_slice(slice_rows.get(name, []))
        if s:
            print(_fmt(name, s))

    print()
    print("Per-class slices (solo drug-only pairs):")
    cls_keys = sorted(
        (k for k in slice_rows if k.startswith("class:")),
        key=lambda k: -len(slice_rows[k]),
    )
    for k in cls_keys:
        print(_fmt(k, _score_slice(slice_rows[k])))

    print()
    print("Per-drug slices (solo drug-only pairs, n>=2):")
    drug_keys = sorted(
        (k for k in slice_rows if k.startswith("drug:")),
        key=lambda k: -len(slice_rows[k]),
    )
    for k in drug_keys:
        if len(slice_rows[k]) < 2:
            continue
        print(_fmt(k, _score_slice(slice_rows[k])))

    # --- A/B vs baseline CSV ---
    if baseline_path:
        print()
        print(f"Regression analysis vs {baseline_path}:")
        base = _load_baseline(baseline_path)
        by_base = {r["_pair_id"]: r for r in base}
        regressed, gained = [], []
        for r in all_rows:
            b = by_base.get(r["_pair_id"])
            if not b:
                continue
            if b["pair_pass"] and not r["pair_pass"]:
                regressed.append((r, b))
            elif r["pair_pass"] and not b["pair_pass"]:
                gained.append((r, b))
        print(f"  Gained  (was fail, now pass): {len(gained)}")
        print(f"  Regressed (was pass, now fail): {len(regressed)}")
        if regressed:
            # Group by drug action for pattern finding
            by_drug = defaultdict(list)
            for r, b in regressed:
                drugs = [a for a in r["_pair"]["actions"] if a.get("type") == "drug"]
                key = drugs[0]["name"] if drugs and len(drugs) == 1 else "non_drug_pair"
                by_drug[key].append(r)
            print("  Regressions by drug name:")
            for k in sorted(by_drug, key=lambda k: -len(by_drug[k])):
                print(f"    {k:<30} n={len(by_drug[k]):>2}")

    # --- Worst-20 drug_only ---
    print()
    print("Worst-20 drug_only pairs:")
    rows = slice_rows.get("drug_only", [])
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
            f"{r['vitals'][v]['predicted']:.0f}(a{r['vitals'][v]['actual']:.0f})"
            for v in worst
        )
        actions = ",".join(r["action_names"])
        tag = f"{r['case_id']}/{r['pair_id']}"
        path = r["_pair"]["before"]["mechanism"]["pathology"]["name"]
        print(f"  [{r['category']:<13}] {tag:<48} path={path:<18} acts=[{actions:<40}] {worst_str}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="transitions")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    run(args.root, args.baseline)
