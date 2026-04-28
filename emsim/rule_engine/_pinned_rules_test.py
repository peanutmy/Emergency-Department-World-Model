"""
Phase 5 pinned drift-rule evaluation.

For each of the 9 rules pinned no-op in Phase 4, try a candidate drift body
under Phase 5 composition and measure the per-pathology net impact (strict
delta + MAE delta) on:
  - pure-wait pairs for that pathology
  - action pairs for that pathology

Run:
    python -m rule_engine._pinned_rules_test [transitions_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rule_engine import pathology_lib
from rule_engine.engine import RuleEngine
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


def _score_pairs_for_path(path_name, root):
    """Run engine over every pair with this pathology; return stats split by slice."""
    engine = RuleEngine()
    stats = {"pure_wait": [], "action": []}
    for cat, case_id, pair in _iter_pairs(root):
        p = pair["before"]["mechanism"]["pathology"]
        if p["name"] != path_name:
            continue
        pred = engine.step(pair["before"], pair["actions"], pair["duration_s"])
        r = evaluate_pair(pair, pred)
        slice_name = "pure_wait" if not pair["actions"] else "action"
        stats[slice_name].append(r)
    return stats


def _summarize(rows):
    if not rows:
        return None
    n = len(rows)
    strict = sum(1 for r in rows if r["pair_pass"])
    mae = {v: sum(r["vitals"][v]["abs_err"] for r in rows) / n for v in VITALS}
    return {"n": n, "strict": strict, "mae": mae}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _with_rule(rule_name, new_body, root):
    """Temporarily swap the registered rule body; score; restore."""
    original = pathology_lib.PATHOLOGY_RULES[rule_name]
    pathology_lib.PATHOLOGY_RULES[rule_name] = new_body
    try:
        stats = _score_pairs_for_path(rule_name, root)
    finally:
        pathology_lib.PATHOLOGY_RULES[rule_name] = original
    return stats


# ======================================================================
# Candidate drift bodies for each pinned rule
# ======================================================================

def _drift_stemi(h, severity, dt_s):
    # State-conditional: only drift when CO/preload already dropped.
    if severity < 0.9:
        return
    if h.CO_index >= 0.9 and h.preload_index >= 0.9:
        return
    dt_min = dt_s / 60.0
    h.chronotropic_drive += 0.15 * dt_min
    h.CO_index = _clip(h.CO_index * (1 - 0.015 * dt_min), 0.1, 4.0)


def _drift_aortic_dissection(h, severity, dt_s):
    # Only O2 drift (the preload drop cost 2 pure-wait pairs).
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.5 * dt_min, 0, 600)


def _drift_adrenal_crisis(h, severity, dt_s):
    # State-conditional: only drift if hypoperfused.
    if severity < 0.9:
        return
    if h.CO_index >= 0.9 and h.SVR_index >= 0.9:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.015 * dt_min), 0.2, 4.0)


def _drift_airway_obstruction(h, severity, dt_s):
    # Lighter magnitude; state-conditional on already-hypoxic.
    if severity < 0.9:
        return
    if h.PaO2_effective >= 70:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)


def _drift_asthma_exacerbation(h, severity, dt_s):
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 1.5 * dt_min, 0, 600)
    h.ventilatory_drive = _clip(h.ventilatory_drive - 0.02 * dt_min, 0, 5.0)


def _drift_septic_shock(h, severity, dt_s):
    # State-conditional on already-hypotensive (SVR_index < 0.8 from residual).
    if severity < 0.9:
        return
    if h.SVR_index >= 0.85:
        return
    dt_min = dt_s / 60.0
    h.SVR_index = _clip(h.SVR_index * (1 - 0.02 * dt_min), 0.2, 4.0)


def _drift_vf_arrest(h, severity, dt_s):
    # post-ROSC decompensation if sinus but hidden state still shocky
    if severity < 0.9:
        return
    dt_min = dt_s / 60.0
    if h.rhythm == "sinus":
        h.CO_index = _clip(h.CO_index * (1 - 0.02 * dt_min), 0.1, 4.0)


def _drift_pea_arrest(h, severity, dt_s):
    # keep no-op as baseline, with state-gated fallback: if sinus (already ROSC),
    # don't drift; if still arrest, no-op too.
    return


def _drift_neonatal_respiratory_distress(h, severity, dt_s):
    # Lighter magnitude; state-conditional — neonatal pairs are mixed
    # (some improve from surfactant phase, some decompensate). Only drift
    # if PaO2 already hypoxic.
    if severity < 0.9:
        return
    if h.PaO2_effective >= 75:
        return
    dt_min = dt_s / 60.0
    h.PaO2_effective = _clip(h.PaO2_effective - 0.4 * dt_min, 0, 600)


CANDIDATES = {
    "stemi": _drift_stemi,
    "aortic_dissection": _drift_aortic_dissection,
    "adrenal_crisis": _drift_adrenal_crisis,
    "airway_obstruction": _drift_airway_obstruction,
    "asthma_exacerbation": _drift_asthma_exacerbation,
    "septic_shock": _drift_septic_shock,
    "vf_arrest": _drift_vf_arrest,
    "pea_arrest": _drift_pea_arrest,
    "neonatal_respiratory_distress": _drift_neonatal_respiratory_distress,
}


def run(root):
    print("Pinned drift-rule Phase 5 re-evaluation\n" + "="*78)
    print(f"{'rule':<32} {'slice':<10} {'n':>3} {'strict_base':>11} {'strict_cand':>11} "
          f"{'d_strict':>9}  HR   BPs   BPd  RR   O2")
    print("-" * 100)
    for name in CANDIDATES:
        # Baseline: keep existing no-op
        engine = RuleEngine()
        base = {"pure_wait": [], "action": []}
        for cat, case_id, pair in _iter_pairs(root):
            p = pair["before"]["mechanism"]["pathology"]
            if p["name"] != name:
                continue
            pred = engine.step(pair["before"], pair["actions"], pair["duration_s"])
            r = evaluate_pair(pair, pred)
            slice_name = "pure_wait" if not pair["actions"] else "action"
            base[slice_name].append(r)

        cand_stats = _with_rule(name, CANDIDATES[name], root)

        for slice_name in ("pure_wait", "action"):
            bs = _summarize(base[slice_name])
            cs = _summarize(cand_stats[slice_name])
            if not bs or not cs:
                continue
            d_strict = cs["strict"] - bs["strict"]
            def _dmae(v):
                return cs["mae"][v] - bs["mae"][v]
            print(f"  {name:<30} {slice_name:<10} {bs['n']:>3} {bs['strict']:>11} "
                  f"{cs['strict']:>11} {d_strict:>+9} "
                  f"{_dmae('HR'):+5.1f} {_dmae('BP_sys'):+5.1f} {_dmae('BP_dia'):+5.1f} "
                  f"{_dmae('RR'):+4.1f} {_dmae('O2Sat'):+5.1f}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    root = sys.argv[1] if len(sys.argv) > 1 else "transitions"
    run(root)
