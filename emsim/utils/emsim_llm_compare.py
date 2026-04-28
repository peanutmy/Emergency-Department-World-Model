"""
Compare LLMEngine (llm_phase0.csv) vs RuleEngine (phase6.csv) on the same 808
pairs. Produces slice tables, an agreement matrix, per-vital MAE, and an
author-marker ceiling check for the atropine drug-only pairs.

Usage:
    python emsim_llm_compare.py \
        --llm llm_phase0.csv --rule phase6.csv \
        --transitions transitions --out llm_compare_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any

from emsim_eval import iter_case_files, VITALS, VITAL_TOLERANCE


KEY_COLS = ("category", "case_id", "pair_id")


def load_csv(path: str) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["category"], row["case_id"], row["pair_id"])
            out[key] = row
    return out


def index_pairs(transitions_dir: str) -> dict[tuple[str, str, str], dict]:
    """Load original pair dicts for slice classification and atropine check."""
    out: dict[tuple[str, str, str], dict] = {}
    for cat, _fname, case in iter_case_files(transitions_dir):
        for pair in case["pairs"]:
            out[(cat, case["case_id"], pair["id"])] = pair
    return out


def row_passes(row: dict) -> bool:
    return str(row.get("pair_pass", "")).lower() == "true"


def row_partial(row: dict) -> float:
    try:
        return float(row.get("partial_score", 0) or 0)
    except Exception:
        return 0.0


def classify_slice(pair: dict) -> list[str]:
    """Return all slice labels a pair belongs to."""
    actions = pair.get("actions", [])
    n = len(actions)
    tags = []
    if n == 0:
        tags.append("pure_wait")
    else:
        has_drug = any(a.get("type") == "drug" for a in actions)
        has_interv = any(a.get("type") == "intervention" for a in actions)
        has_defib = any(a.get("type") == "intervention" and a.get("name") == "defibrillate" for a in actions)
        if has_defib:
            tags.append("defib")
        if has_drug and not has_interv:
            tags.append("drug_only")
        if has_interv and not has_drug and not has_defib:
            tags.append("interv_only")
        if has_interv and not has_defib:
            tags.append("action_non_defib")
        if has_drug and has_interv:
            tags.append("mixed")
    # Arrest subset: pathology name startswith arrest-ish OR CPR_active true in before.
    before = pair.get("before", {})
    path_name = (before.get("mechanism", {}).get("pathology", {}).get("name") or "").lower()
    iv = before.get("interventions", {})
    if "arrest" in path_name or iv.get("CPR_active") or before.get("vitals", {}).get("HR") == 0:
        tags.append("arrest")
    return tags


def mae_for(rows: list[dict], vital: str) -> float:
    vals = [float(r.get(f"{vital}_abs_err") or 0) for r in rows if r.get(f"{vital}_abs_err") not in (None, "", "None")]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def summarize_slice(rows: list[dict]) -> dict:
    n = len(rows)
    passed = sum(1 for r in rows if row_passes(r))
    partial = sum(row_partial(r) for r in rows) / max(n, 1)
    return {
        "n": n,
        "pass": passed,
        "pct": (100 * passed / n) if n else 0.0,
        "partial": partial,
        "mae": {v: mae_for(rows, v) for v in VITALS},
    }


def fmt_slice(label: str, s: dict) -> str:
    return (f"{label:<22} n={s['n']:>3}  pass={s['pass']:>3}/{s['n']:<3} "
            f"({s['pct']:5.1f}%)  partial={s['partial']:.3f}  "
            f"HR={s['mae']['HR']:5.2f} BP_s={s['mae']['BP_sys']:5.2f} "
            f"BP_d={s['mae']['BP_dia']:5.2f} RR={s['mae']['RR']:4.2f} "
            f"O2={s['mae']['O2Sat']:5.2f} T={s['mae']['T']:.2f}")


def slice_table(llm_rows_by_key, rule_rows_by_key, pairs_by_key):
    slices: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    # Compute overall + per-slice buckets.
    for key, pair in pairs_by_key.items():
        if key not in llm_rows_by_key or key not in rule_rows_by_key:
            continue
        l = llm_rows_by_key[key]
        r = rule_rows_by_key[key]
        slices["OVERALL"].append((l, r))
        for tag in classify_slice(pair):
            slices[tag].append((l, r))

    order = ["OVERALL", "pure_wait", "drug_only", "interv_only",
             "action_non_defib", "defib", "mixed", "arrest"]

    lines = []
    lines.append(f"{'slice':<22} {'eng':<5} {'n':>4} {'pass':<12} {'partial':>7}  "
                 f"{'HR':>5} {'BPs':>5} {'BPd':>5} {'RR':>5} {'O2':>5} {'T':>5}")
    for name in order:
        if name not in slices:
            continue
        pairs = slices[name]
        lls = [p[0] for p in pairs]
        rls = [p[1] for p in pairs]
        for label, rows in (("LLM", lls), ("RULE", rls)):
            s = summarize_slice(rows)
            lines.append(
                f"{name:<22} {label:<5} {s['n']:>4} "
                f"{s['pass']:>3}/{s['n']:<3} ({s['pct']:5.1f}%) {s['partial']:.3f}  "
                f"{s['mae']['HR']:>5.2f} {s['mae']['BP_sys']:>5.2f} "
                f"{s['mae']['BP_dia']:>5.2f} {s['mae']['RR']:>5.2f} "
                f"{s['mae']['O2Sat']:>5.2f} {s['mae']['T']:>5.2f}"
            )
    return "\n".join(lines), slices


def agreement_matrix(slices_overall):
    both_pass = rule_only = llm_only = both_fail = 0
    for l, r in slices_overall:
        lp, rp = row_passes(l), row_passes(r)
        if lp and rp:
            both_pass += 1
        elif rp and not lp:
            rule_only += 1
        elif lp and not rp:
            llm_only += 1
        else:
            both_fail += 1
    return both_pass, rule_only, llm_only, both_fail


def per_category(llm_by_key, rule_by_key):
    by_cat_llm = defaultdict(list)
    by_cat_rule = defaultdict(list)
    for key, l in llm_by_key.items():
        if key not in rule_by_key:
            continue
        cat = l["category"]
        by_cat_llm[cat].append(l)
        by_cat_rule[cat].append(rule_by_key[key])
    return by_cat_llm, by_cat_rule


def atropine_ceiling(pairs_by_key, llm_by_key, rule_by_key):
    """For drug-only atropine pairs, compare what each engine predicts for HR
    against the authored ground truth. Rule-based passes these by near-identity
    (marker-dominant). If LLM predicts large HR delta and fails, that confirms
    the 22.9% ceiling is corpus-structural, not engine-architectural."""
    rows = []
    for key, pair in pairs_by_key.items():
        actions = pair.get("actions", [])
        if len(actions) != 1:
            continue
        a = actions[0]
        if not (a.get("type") == "drug" and a.get("name") == "atropine"):
            continue
        if key not in llm_by_key or key not in rule_by_key:
            continue
        l = llm_by_key[key]
        r = rule_by_key[key]
        hr_before = float(pair["before"]["vitals"]["HR"])
        hr_after = float(pair["after"]["vitals"]["HR"])
        rows.append({
            "case": f"{pair['before']['mechanism']['pathology']['name']}/{key[2]}",
            "hr_before": hr_before,
            "hr_after": hr_after,
            "hr_llm": float(l.get("HR_predicted") or 0),
            "hr_rule": float(r.get("HR_predicted") or 0),
            "llm_pass": row_passes(l),
            "rule_pass": row_passes(r),
        })
    return rows


def worst20(slices_overall, n=20):
    """Ranked by tolerance-normalized vital error sum, for LLM."""
    scored = []
    for l, r in slices_overall:
        def _sum_norm(row):
            s = 0.0
            for v in VITALS:
                try:
                    ae = float(row.get(f"{v}_abs_err") or 0)
                except Exception:
                    ae = 0.0
                s += ae / VITAL_TOLERANCE[v]
            return s
        scored.append((_sum_norm(l), l, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:n]


def format_agreement(both_p, rule_only, llm_only, both_f):
    total = both_p + rule_only + llm_only + both_f
    def pct(x): return f"{100*x/max(total,1):5.1f}%"
    return (
        f"|                 | rule pass | rule fail |\n"
        f"|-----------------|-----------|-----------|\n"
        f"| LLM pass        | {both_p:3d} ({pct(both_p)}) | {llm_only:3d} ({pct(llm_only)}) |\n"
        f"| LLM fail        | {rule_only:3d} ({pct(rule_only)}) | {both_f:3d} ({pct(both_f)}) |\n"
        f"\nTotal: {total}. LLM-only wins: {llm_only}, rule-only wins: {rule_only}."
    )


def disagreement_samples(slices_overall, pairs_by_key, llm_by_key, rule_by_key,
                         want_llm_only=10, want_rule_only=10):
    llm_only_samples = []
    rule_only_samples = []
    for l, r in slices_overall:
        key = (l["category"], l["case_id"], l["pair_id"])
        lp, rp = row_passes(l), row_passes(r)
        if lp and not rp and len(llm_only_samples) < want_llm_only:
            llm_only_samples.append((key, l, r, pairs_by_key[key]))
        elif rp and not lp and len(rule_only_samples) < want_rule_only:
            rule_only_samples.append((key, l, r, pairs_by_key[key]))
    return llm_only_samples, rule_only_samples


def format_sample_pairs(title: str, samples, include_reasoning_from_cache=True):
    from emsim_llm_engine import LLMEngine
    eng = LLMEngine(cache_enabled=True)
    lines = [f"### {title}\n"]
    for (key, l, r, pair) in samples:
        before = pair["before"]
        after = pair["after"]
        actions_repr = ", ".join(
            a.get("name", "?") + (f"({a.get('dose','')}{a.get('unit','')})"
                                  if a.get("type") == "drug" else "")
            for a in pair["actions"]
        ) or "(no actions)"
        path = before["mechanism"]["pathology"]
        lines.append(f"**{key[1]}/{key[2]}** — pathology={path['name']}/{path['severity']}, "
                     f"duration={pair['duration_s']}s, actions=[{actions_repr}]")
        for v in ("HR", "BP_sys", "BP_dia", "RR", "O2Sat"):
            lines.append(f"  {v}: {before['vitals'][v]} → actual {after['vitals'][v]}, "
                         f"LLM={l.get(v+'_predicted')}, RULE={r.get(v+'_predicted')}")
        if include_reasoning_from_cache:
            cache_key = eng._cache_key(before, pair["actions"], pair["duration_s"])
            cached = eng._load_cache(cache_key)
            if cached:
                r_text = cached.get("reasoning") or "(no reasoning)"
                lines.append(f"  LLM reasoning: _{r_text}_")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="llm_phase0.csv")
    ap.add_argument("--rule", default="phase6.csv")
    ap.add_argument("--transitions", default="transitions")
    ap.add_argument("--out", default="llm_compare_report.md")
    args = ap.parse_args()

    llm_by_key = load_csv(args.llm)
    rule_by_key = load_csv(args.rule)
    pairs_by_key = index_pairs(args.transitions)

    # Engine stats from cache.
    cache_files = 0
    llm_failure_markers = 0
    if os.path.isdir(os.path.join(os.path.dirname(__file__), ".cache", "llm")):
        cdir = os.path.join(os.path.dirname(__file__), ".cache", "llm")
        for fname in os.listdir(cdir):
            if fname.endswith(".json"):
                cache_files += 1
                try:
                    with open(os.path.join(cdir, fname), encoding="utf-8") as f:
                        d = json.load(f)
                    if d.get("__failed__"):
                        llm_failure_markers += 1
                except Exception:
                    pass

    table_str, slices = slice_table(llm_by_key, rule_by_key, pairs_by_key)
    overall_pairs = slices["OVERALL"]
    both_p, rule_only, llm_only, both_f = agreement_matrix(overall_pairs)

    # Author-marker: atropine drug-only pairs.
    atropine = atropine_ceiling(pairs_by_key, llm_by_key, rule_by_key)

    # Worst-20 for LLM
    worst = worst20(overall_pairs, 20)

    # Per-category
    by_cat_llm, by_cat_rule = per_category(llm_by_key, rule_by_key)

    llm_only_samples, rule_only_samples = disagreement_samples(
        overall_pairs, pairs_by_key, llm_by_key, rule_by_key
    )

    lines = []
    lines.append("# EMSim LLM Phase-0 Comparison Report\n")
    lines.append("_Generated by `emsim_llm_compare.py`._\n")

    lines.append("## 1. Cache / schema health\n")
    lines.append(f"- Cache entries (unique keys): **{cache_files}**")
    lines.append(f"- LLM call failures cached as identity: **{llm_failure_markers}**")
    lines.append(f"- LLM CSV rows: {len(llm_by_key)}, Rule CSV rows: {len(rule_by_key)}\n")

    lines.append("## 2. Slice comparison — LLM vs RULE\n")
    lines.append("```")
    lines.append(table_str)
    lines.append("```\n")

    lines.append("## 3. Agreement matrix (strict pair_pass, overall)\n")
    lines.append(format_agreement(both_p, rule_only, llm_only, both_f))
    lines.append("")

    lines.append("## 4. Author-marker ceiling check — atropine drug-only pairs\n")
    lines.append("If LLM also predicts large HR jumps, the 22.9% ceiling is corpus-structural.")
    lines.append("")
    lines.append("| Case | HR before | HR authored | HR rule | HR LLM | rule pass | LLM pass |")
    lines.append("|------|-----------|-------------|---------|--------|-----------|----------|")
    for row in atropine:
        lines.append(
            f"| {row['case']} | {row['hr_before']:.0f} | {row['hr_after']:.0f} | "
            f"{row['hr_rule']:.0f} | {row['hr_llm']:.0f} | "
            f"{'✓' if row['rule_pass'] else '✗'} | {'✓' if row['llm_pass'] else '✗'} |"
        )
    lines.append("")
    if atropine:
        big_llm_jumps = sum(1 for a in atropine
                            if abs(a["hr_llm"] - a["hr_before"]) > 15
                            and abs(a["hr_after"] - a["hr_before"]) < 10)
        lines.append(f"**Big LLM HR jumps on author-stable atropine pairs: "
                     f"{big_llm_jumps}/{len(atropine)}**")

    lines.append("\n## 5. Per-category pair_pass\n")
    lines.append("| Category | LLM pass | Rule pass |")
    lines.append("|----------|----------|-----------|")
    for cat in sorted(by_cat_llm):
        lls = by_cat_llm[cat]
        rls = by_cat_rule[cat]
        lp = sum(1 for r in lls if row_passes(r))
        rp = sum(1 for r in rls if row_passes(r))
        lines.append(f"| {cat} | {lp}/{len(lls)} ({100*lp/len(lls):.1f}%) "
                     f"| {rp}/{len(rls)} ({100*rp/len(rls):.1f}%) |")

    lines.append("\n## 6. Worst-20 LLM pairs (by tol-normalized vital error)\n")
    lines.append("| case / pair | LLM partial | RULE partial | LLM pass | RULE pass |")
    lines.append("|-------------|-------------|--------------|----------|-----------|")
    for _score, l, r in worst:
        lines.append(
            f"| {l['case_id']}/{l['pair_id']} | "
            f"{row_partial(l):.3f} | {row_partial(r):.3f} | "
            f"{'✓' if row_passes(l) else '✗'} | {'✓' if row_passes(r) else '✗'} |"
        )

    lines.append("\n## 7. Sample disagreement pairs\n")
    lines.append(format_sample_pairs("LLM-only pass (10)", llm_only_samples))
    lines.append(format_sample_pairs("Rule-only pass (10)", rule_only_samples))

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"Agreement: both={both_p}, rule_only={rule_only}, llm_only={llm_only}, both_fail={both_f}")


if __name__ == "__main__":
    main()
