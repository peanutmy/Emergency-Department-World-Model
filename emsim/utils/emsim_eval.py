"""
EMSim Physiology Engine — evaluation harness.

Loads all transition pair JSONs, runs a provided engine on each pair, and
scores predictions against authored ground truth using tolerance bands.

Usage:
    python emsim_eval.py --transitions D:/wmed/emsim/transitions
    python emsim_eval.py --transitions ./transitions --export results.csv
    python emsim_eval.py --transitions ./transitions --worst 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any


# ======================================================================
# Tolerance configuration
# ======================================================================
# Absolute-error bands: |predicted - actual| <= tolerance → "within tolerance".
# Calibrated so author-level quantization noise is absorbed; tighten later.
VITAL_TOLERANCE = {
    "HR":     10,
    "BP_sys": 15,
    "BP_dia": 15,
    "RR":     4,
    "O2Sat":  3,
    "T":      0.3,
}

# Direction dead-zone: change smaller than this is treated as "stable".
# Direction-correctness is evaluated separately from tolerance, to catch
# engines that are numerically close but move the wrong way.
VITAL_DIR_DEADZONE = {
    "HR":     3,
    "BP_sys": 5,
    "BP_dia": 5,
    "RR":     1,
    "O2Sat":  1,
    "T":      0.1,
}

# Boolean intervention flags — exact match required.
INTERV_BOOL_KEYS = [
    "airway", "intubated", "CPR_active", "pacing_active",
    "warming_active", "cooling_active",
    "needle_decompression", "chest_tube", "pericardiocentesis",
]

# Categorical (enum-or-null) — exact match required.
INTERV_CAT_KEYS = ["O2_device", "fluid_type"]

# Numeric intervention settings — with tolerance.
# 0 = exact match required.
INTERV_NUM_TOLERANCE = {
    "PEEP":              0,
    "FiO2":              0.05,
    "vent_rate":         2,
    "vent_TV_ml":        50,
    "defib_last_J":      0,
    "pacing_rate":       5,
    "fluids_rate_ml_hr": 25,
}

VITALS = list(VITAL_TOLERANCE.keys())

# Rhythm vocabulary (mirrors hidden_state.RHYTHMS). Evaluated independently
# from vitals — see `evaluate_pair` for the rhythm_pair_pass channel.
RHYTHMS = ("sinus", "SVT", "VT", "VF", "asystole", "PEA", "bradycardia")


# ======================================================================
# Per-pair scoring
# ======================================================================

def _direction(delta: float, deadzone: float) -> str:
    if abs(delta) <= deadzone:
        return "stable"
    return "up" if delta > 0 else "down"


def _nullable_num_match(p, a, tol) -> bool:
    """Match for numeric fields that may be None."""
    if p is None and a is None:
        return True
    if p is None or a is None:
        return False
    return abs(p - a) <= tol


def evaluate_pair(pair: dict, predicted: dict) -> dict:
    """Score a single predicted `after` against the authored `after`."""
    before = pair["before"]
    actual = pair["after"]

    result: dict[str, Any] = {
        "vitals": {},
        "interv_bool": {},
        "interv_cat": {},
        "interv_num": {},
        "mechanism": {},
    }

    # --- Vitals: tolerance + direction ---
    for v in VITALS:
        tol = VITAL_TOLERANCE[v]
        dz  = VITAL_DIR_DEADZONE[v]
        pv  = predicted["vitals"][v]
        av  = actual["vitals"][v]
        bv  = before["vitals"][v]
        abs_err = abs(pv - av)
        result["vitals"][v] = {
            "before":     bv,
            "actual":     av,
            "predicted":  pv,
            "abs_err":    abs_err,
            "within_tol": abs_err <= tol,
            "dir_actual": _direction(av - bv, dz),
            "dir_pred":   _direction(pv - bv, dz),
        }
        result["vitals"][v]["dir_match"] = (
            result["vitals"][v]["dir_actual"] == result["vitals"][v]["dir_pred"]
        )

    # --- Intervention flags ---
    p_iv = predicted["interventions"]
    a_iv = actual["interventions"]
    for k in INTERV_BOOL_KEYS:
        result["interv_bool"][k] = {
            "predicted": p_iv.get(k),
            "actual":    a_iv.get(k),
            "match":     p_iv.get(k) == a_iv.get(k),
        }
    for k in INTERV_CAT_KEYS:
        result["interv_cat"][k] = {
            "predicted": p_iv.get(k),
            "actual":    a_iv.get(k),
            "match":     p_iv.get(k) == a_iv.get(k),
        }
    for k, tol in INTERV_NUM_TOLERANCE.items():
        pn, an = p_iv.get(k), a_iv.get(k)
        result["interv_num"][k] = {
            "predicted": pn,
            "actual":    an,
            "match":     _nullable_num_match(pn, an, tol),
        }

    # --- Mechanism (pathology name + severity) ---
    p_path = predicted["mechanism"]["pathology"]
    a_path = actual["mechanism"]["pathology"]
    result["mechanism"] = {
        "name_match":     p_path["name"]     == a_path["name"],
        "severity_match": p_path["severity"] == a_path["severity"],
    }

    # --- Rhythm (independent channel, optional) ---
    # Evaluated separately from vitals so the rhythm_pair_pass signal does
    # not contaminate the legacy vitals/interv/mech pair_pass. If either
    # ground truth or prediction lacks `mechanism.rhythm`, the comparison
    # is skipped (`match = None`) — neither pass nor fail. This preserves
    # backward compatibility with pre-rhythm pairs.
    p_rhythm = predicted.get("mechanism", {}).get("rhythm")
    a_rhythm = actual.get("mechanism", {}).get("rhythm")
    if p_rhythm is None or a_rhythm is None:
        rhythm_match: bool | None = None
    else:
        rhythm_match = (p_rhythm == a_rhythm)
    result["rhythm"] = {
        "predicted": p_rhythm,
        "actual":    a_rhythm,
        "match":     rhythm_match,
    }

    # --- Aggregated flags ---
    # Drug comparison removed 2026-04-19: drugs no longer live in state[]; they
    # appear only in pair.actions[] (which the engine reads, not emits).
    result["all_vitals_in_tol"]    = all(r["within_tol"] for r in result["vitals"].values())
    result["all_dirs_match"]       = all(r["dir_match"]  for r in result["vitals"].values())
    result["all_interv_match"]     = (
        all(r["match"] for r in result["interv_bool"].values()) and
        all(r["match"] for r in result["interv_cat"].values())  and
        all(r["match"] for r in result["interv_num"].values())
    )
    result["mech_match"]           = result["mechanism"]["name_match"] and result["mechanism"]["severity_match"]
    # Pure vitals channel — 6 vitals within tolerance AND moving the right
    # direction. Independent of interventions and mechanism.
    result["vitals_pair_pass"] = result["all_vitals_in_tol"] and result["all_dirs_match"]
    # Soft vitals signal — counts how many vitals fully pass (both
    # within_tol AND dir_match). Useful when strict 6/6 is too coarse
    # to track incremental engine progress; the dir_match deadzones are
    # tight enough that single-axis drift improvements rarely flip the
    # strict pass but show up here. `vitals_partial_score` is over the
    # 12 underlying checks (within_tol + dir_match per vital), bounded [0, 1].
    result["vitals_pass_count"] = sum(
        1 for r in result["vitals"].values()
        if r["within_tol"] and r["dir_match"]
    )
    n_within = sum(1 for r in result["vitals"].values() if r["within_tol"])
    n_dir    = sum(1 for r in result["vitals"].values() if r["dir_match"])
    result["vitals_partial_score"] = (n_within + n_dir) / (len(VITALS) * 2)
    # `pair_pass` (legacy) — covers vitals/interv/mech only. Rhythm lives on
    # its own channel below; combine via `combined_pair_pass` if/when needed.
    result["pair_pass"] = (
        result["all_vitals_in_tol"] and
        result["all_dirs_match"]    and
        result["all_interv_match"]  and
        result["mech_match"]
    )
    # Independent rhythm channel: True/False/None (None = skipped).
    result["rhythm_pair_pass"] = rhythm_match
    # Combined view (only meaningful when rhythm is present):
    #   None → use legacy pair_pass; True/False → AND with vitals pair_pass.
    if rhythm_match is None:
        result["combined_pair_pass"] = result["pair_pass"]
    else:
        result["combined_pair_pass"] = result["pair_pass"] and rhythm_match

    # Partial-credit score in [0, 1] — useful as a continuous signal.
    # NOTE: rhythm is intentionally NOT included in partial_score so existing
    # benchmark numbers stay comparable. Rhythm is reported via the dedicated
    # rhythm_pair_pass channel.
    n_checks = len(VITALS) * 2 + len(INTERV_BOOL_KEYS) + len(INTERV_CAT_KEYS) + len(INTERV_NUM_TOLERANCE) + 2
    n_pass = (
        sum(r["within_tol"] for r in result["vitals"].values()) +
        sum(r["dir_match"]  for r in result["vitals"].values()) +
        sum(r["match"] for r in result["interv_bool"].values()) +
        sum(r["match"] for r in result["interv_cat"].values())  +
        sum(r["match"] for r in result["interv_num"].values())  +
        int(result["mechanism"]["name_match"]) +
        int(result["mechanism"]["severity_match"])
    )
    result["partial_score"] = n_pass / n_checks

    return result


# ======================================================================
# Harness
# ======================================================================

def iter_case_files(transitions_dir: str, categories: set[str] | None = None):
    """Yield (category, filename, case_dict) for every JSON under transitions/.

    If `categories` is provided, only those category folders are visited.
    """
    for cat in sorted(os.listdir(transitions_dir)):
        if categories is not None and cat not in categories:
            continue
        cat_path = os.path.join(transitions_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".json") or fname == "schema.json":
                continue
            with open(os.path.join(cat_path, fname), "r", encoding="utf-8") as f:
                case = json.load(f)
            yield cat, fname, case


def run_eval(engine, transitions_dir: str, categories: set[str] | None = None) -> list[dict]:
    """Run engine over every pair and collect per-pair results."""
    results = []
    for cat, fname, case in iter_case_files(transitions_dir, categories):
        for pair in case["pairs"]:
            try:
                predicted = engine.step(
                    pair["before"],
                    pair["actions"],
                    pair["duration_s"],
                )
                r = evaluate_pair(pair, predicted)
                r["error"] = None
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}", "pair_pass": False, "partial_score": 0.0}
            r["category"] = cat
            r["case_id"]  = case["case_id"]
            r["pair_id"]  = pair["id"]
            r["file"]     = fname
            r["n_actions"]  = len(pair["actions"])
            r["duration_s"] = pair["duration_s"]
            results.append(r)
    return results


# ======================================================================
# Reporting
# ======================================================================

def _fmt_pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}%" if den else "  n/a"


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("No pairs found.")
        return

    errored = [r for r in results if r.get("error")]
    scored  = [r for r in results if not r.get("error")]

    # Two independent channels — vitals and rhythm — reported separately.
    n_vitals_pass = sum(1 for r in scored if r["vitals_pair_pass"])
    mean_partial  = (sum(r["vitals_partial_score"] for r in scored)
                     / max(len(scored), 1))

    rhythm_evaluated = [r for r in scored if r["rhythm_pair_pass"] is not None]
    n_rhythm_pass    = sum(1 for r in rhythm_evaluated if r["rhythm_pair_pass"])

    print("=" * 72)
    print("EMSim Physiology Engine — Evaluation (vitals + rhythm channels)")
    print("=" * 72)
    print(f"Total pairs             : {n}")
    print(f"Engine errors           : {len(errored)}")
    print(f"Vitals pair pass        : {n_vitals_pass}/{len(scored)}  "
          f"({_fmt_pct(n_vitals_pass, len(scored))})  (strict: 6 vitals within tol AND right direction)")
    print(f"Vitals partial score    : {mean_partial:.3f}     "
          f"(mean over pairs of fraction of 12 checks passing — within_tol + dir_match per vital)")

    # Distribution: how many pairs land at each "vitals fully pass" count (0..6).
    from collections import Counter as _Counter
    pass_count_dist = _Counter(r["vitals_pass_count"] for r in scored)
    counts_str = "  ".join(
        f"{k}/6:{pass_count_dist.get(k, 0):>3}" for k in range(6, -1, -1)
    )
    print(f"Vitals fully-pass dist  : {counts_str}")

    if rhythm_evaluated:
        print(f"Rhythm pair pass        : {n_rhythm_pass}/{len(rhythm_evaluated)}  "
              f"({_fmt_pct(n_rhythm_pass, len(rhythm_evaluated))})  "
              f"(skipped {len(scored) - len(rhythm_evaluated)} pairs lacking ground-truth rhythm)")
    else:
        print(f"Rhythm pair pass        : n/a  (no pairs have ground-truth rhythm yet)")

    # --- Per-vital (vitals-channel detail) ---
    print()
    print("Per-vital metrics (over scored pairs):")
    print(f"  {'vital':<8}  {'within_tol':>10}  {'dir_match':>10}  {'MAE':>8}")
    for v in VITALS:
        within = sum(1 for r in scored if r["vitals"][v]["within_tol"])
        dirm   = sum(1 for r in scored if r["vitals"][v]["dir_match"])
        mae    = sum(r["vitals"][v]["abs_err"] for r in scored) / max(len(scored), 1)
        print(f"  {v:<8}  {_fmt_pct(within, len(scored)):>10}  {_fmt_pct(dirm, len(scored)):>10}  {mae:>8.2f}")

    # --- Per category (both channels side-by-side) ---
    print()
    print("Per-category pass rates  (vitals  /  rhythm):")
    by_cat: dict[str, list] = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        v_pass = sum(1 for r in rs if r["vitals_pair_pass"])
        r_eval = [r for r in rs if r["rhythm_pair_pass"] is not None]
        r_pass = sum(1 for r in r_eval if r["rhythm_pair_pass"])
        rhythm_str = (f"{r_pass:>3}/{len(r_eval):<3} {_fmt_pct(r_pass, len(r_eval))}"
                      if r_eval else "      n/a    ")
        print(f"  {cat:<18}  {v_pass:>3}/{len(rs):<3} {_fmt_pct(v_pass, len(rs))}   "
              f"{rhythm_str}")

    # --- Pure-wait vs action pairs (vitals channel) ---
    print()
    wait   = [r for r in scored if r["n_actions"] == 0]
    acted  = [r for r in scored if r["n_actions"] > 0]
    if wait:
        p = sum(1 for r in wait if r["vitals_pair_pass"])
        print(f"Pure-wait pairs    (actions=0): {p}/{len(wait)}  {_fmt_pct(p, len(wait))}  (vitals)")
    if acted:
        p = sum(1 for r in acted if r["vitals_pair_pass"])
        print(f"Action pairs     (actions>=1): {p}/{len(acted)}  {_fmt_pct(p, len(acted))}  (vitals)")


def print_worst(results: list[dict], n: int = 10) -> None:
    scored = [r for r in results if not r.get("error")]
    ranked = sorted(
        scored,
        key=lambda r: -sum(v["abs_err"] / VITAL_TOLERANCE[name] for name, v in r["vitals"].items()),
    )
    print()
    print(f"Top {n} worst-predicted pairs (by tolerance-normalized vital error):")
    print(f"  {'category':<14} {'case / pair':<50} {'score':>6}  worst vitals")
    for r in ranked[:n]:
        worst = sorted(
            r["vitals"].items(),
            key=lambda kv: -kv[1]["abs_err"] / VITAL_TOLERANCE[kv[0]],
        )[:3]
        worst_str = ", ".join(
            f"{k}:{v['before']:.0f}->{v['predicted']:.0f} (actual {v['actual']:.0f})"
            for k, v in worst
        )
        tag = f"{r['case_id']}/{r['pair_id']}"
        print(f"  {r['category']:<14} {tag:<50} {r['partial_score']:>6.2f}  {worst_str}")


def export_csv(results: list[dict], path: str) -> None:
    """Flatten per-pair results into a CSV row per pair for deeper analysis."""
    fieldnames = [
        "category", "case_id", "pair_id", "file",
        "n_actions", "duration_s",
        # Two independent channels + soft vitals signal.
        "vitals_pair_pass", "vitals_pass_count", "vitals_partial_score",
        "rhythm_pair_pass", "rhythm_predicted", "rhythm_actual",
        # Legacy aggregates kept for backward-compat with older tooling.
        "pair_pass", "partial_score",
        "all_vitals_in_tol", "all_dirs_match", "all_interv_match",
        "mech_match", "combined_pair_pass",
        "error",
    ]
    for v in VITALS:
        fieldnames += [f"{v}_before", f"{v}_actual", f"{v}_predicted",
                       f"{v}_abs_err", f"{v}_within_tol", f"{v}_dir_match"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in [
                "category", "case_id", "pair_id", "file", "n_actions", "duration_s",
                "vitals_pair_pass", "vitals_pass_count", "vitals_partial_score",
                "pair_pass", "partial_score",
                "all_vitals_in_tol", "all_dirs_match", "all_interv_match",
                "mech_match", "error",
            ]}
            if not r.get("error"):
                for v in VITALS:
                    row[f"{v}_before"]     = r["vitals"][v]["before"]
                    row[f"{v}_actual"]     = r["vitals"][v]["actual"]
                    row[f"{v}_predicted"]  = r["vitals"][v]["predicted"]
                    row[f"{v}_abs_err"]    = r["vitals"][v]["abs_err"]
                    row[f"{v}_within_tol"] = r["vitals"][v]["within_tol"]
                    row[f"{v}_dir_match"]  = r["vitals"][v]["dir_match"]
                # Independent rhythm channel (may be None if either side lacks
                # ground-truth rhythm — preserved as empty in CSV).
                rhythm = r.get("rhythm", {})
                row["rhythm_predicted"]   = rhythm.get("predicted")
                row["rhythm_actual"]      = rhythm.get("actual")
                row["rhythm_pair_pass"]   = r.get("rhythm_pair_pass")
                row["combined_pair_pass"] = r.get("combined_pair_pass")
            w.writerow(row)


# ======================================================================
# CLI
# ======================================================================

def _load_engine(spec: str):
    """
    Load an engine by spec "module:ClassName". Default = emsim_engine:IdentityEngine.
    """
    mod_name, cls_name = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def main():
    ap = argparse.ArgumentParser(description="EMSim physiology engine evaluation harness")
    ap.add_argument("--transitions", required=True,
                    help="Path to transitions/ directory containing <category>/*.json")
    ap.add_argument("--engine", default="emsim_engine:IdentityEngine",
                    help="Engine to load, format 'module:Class'. Default: IdentityEngine baseline.")
    ap.add_argument("--export",  default=None, help="Optional CSV export path.")
    ap.add_argument("--worst",   type=int, default=10,
                    help="Show top-N worst-predicted pairs (0 to skip).")
    ap.add_argument("--category", action="append", default=None,
                    help="Restrict to one or more category folders (repeatable). "
                         "Default: all categories.")
    args = ap.parse_args()

    engine = _load_engine(args.engine)
    categories = set(args.category) if args.category else None
    results = run_eval(engine, args.transitions, categories=categories)
    print_summary(results)
    if args.worst:
        print_worst(results, args.worst)
    if args.export:
        export_csv(results, args.export)
        print(f"\nExported per-pair CSV -> {args.export}")


if __name__ == "__main__":
    main()
