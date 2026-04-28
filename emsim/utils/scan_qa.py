"""
Systematic QA scan across all transition-pair JSONs to find:

  A) Split-should-merge: action pair (duration=0) followed by null pair (duration>0)
     with no additional intervention change — typically a delayed-effect action split
     incorrectly into two pairs.

  B) Missing CPR: dramatic HR recovery (e.g., 0 -> >=60, or <50 -> >=80) without
     CPR_active=true anywhere in the pair. Scenario likely had CPR implicit.

  C) Big vital change with empty actions AND short duration (<60s) and no
     persistent intervention active — suspicious.

  D) Severity mismatch: pathology.severity stays the same despite large vital
     deterioration or recovery.

  E) Rhythm-change-like pattern: HR very low (≤40) -> normal range, no pacing /
     CPR / defib flagged.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\wmed\emsim\transitions")

findings = defaultdict(list)  # key -> list of (file, pair_id, note)


def hr(state): return state["vitals"]["HR"]
def sbp(state): return state["vitals"]["BP_sys"]
def spo2(state): return state["vitals"]["O2Sat"]
def cpr_on(state): return state["interventions"]["CPR_active"]
def pacing_on(state): return state["interventions"]["pacing_active"]
def has_defib(state): return state["interventions"]["defib_last_J"] is not None
def severity(state): return state["mechanism"]["pathology"]["severity"]


def scan_case(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    case = str(path.relative_to(ROOT))

    # Pattern A: action-pair (duration=0) + null-pair (duration>0) consecutive,
    # where action pair's intervention change is small/trivial
    for i in range(len(pairs) - 1):
        a, b = pairs[i], pairs[i + 1]
        if (len(a["actions"]) >= 1
            and a["duration_s"] == 0
            and len(b["actions"]) == 0
            and b["duration_s"] > 0
            and a["after"] == b["before"]):
            # Likely a delayed-effect split
            action_names = [x.get("name", x["type"]) for x in a["actions"]]
            # Flag if the vital change is mostly in pair B (delayed)
            a_delta = abs(spo2(a["after"]) - spo2(a["before"])) + abs(hr(a["after"]) - hr(a["before"]))
            b_delta = abs(spo2(b["after"]) - spo2(b["before"])) + abs(hr(b["after"]) - hr(b["before"]))
            if b_delta > 0:
                findings["A_split_merge"].append(
                    (case, f"{a['id']}+{b['id']}",
                     f"actions={action_names}, durA=0 durB={b['duration_s']}, "
                     f"dA={a_delta:.0f} dB={b_delta:.0f}")
                )

    # Pattern B: HR recovery suggests CPR but CPR_active not set
    for p in pairs:
        hr_before, hr_after = hr(p["before"]), hr(p["after"])
        # Arrest-like: HR very low or zero before
        if hr_before <= 40 and hr_after >= 60:
            cpr_any = cpr_on(p["before"]) or cpr_on(p["after"])
            pacing_any = pacing_on(p["before"]) or pacing_on(p["after"])
            defib_any = has_defib(p["before"]) or has_defib(p["after"])
            action_names = [x.get("name", x["type"]) for x in p["actions"]]
            explicit_cpr = any("CPR" in n or "cpr" in n for n in action_names)
            explicit_pacing = any("pacing" in n.lower() for n in action_names)
            explicit_defib = any("defib" in n.lower() or "cardiovers" in n.lower() for n in action_names)
            if not (cpr_any or pacing_any or defib_any or explicit_cpr or explicit_pacing or explicit_defib):
                findings["B_hr_recovery_no_cpr"].append(
                    (case, p["id"],
                     f"HR {hr_before}->{hr_after}, actions={action_names}, "
                     f"no CPR/pacing/defib anywhere")
                )

    # Pattern C: Big vital change with empty actions AND no persistent intervention
    # change and short duration
    for p in pairs:
        if len(p["actions"]) > 0:
            continue
        if p["duration_s"] > 60:
            continue  # long waits can have physiology drift
        dhr = abs(hr(p["after"]) - hr(p["before"]))
        dsbp = abs(sbp(p["after"]) - sbp(p["before"]))
        dspo2 = abs(spo2(p["after"]) - spo2(p["before"]))
        if dhr >= 20 or dsbp >= 30 or dspo2 >= 8:
            findings["C_big_change_no_cause"].append(
                (case, p["id"],
                 f"empty actions, dur={p['duration_s']}s, dHR={dhr} dSBP={dsbp} dSpO2={dspo2}")
            )

    # Pattern D: Severity unchanged despite arrest (HR=0 or BP=0)
    for p in pairs:
        before_arrest = hr(p["before"]) == 0 or sbp(p["before"]) == 0
        after_arrest = hr(p["after"]) == 0 or sbp(p["after"]) == 0
        sev_b = severity(p["before"])
        sev_a = severity(p["after"])
        if before_arrest and sev_b != "severe":
            findings["D_severity_arrest_mismatch"].append(
                (case, p["id"], f"before is arrest but severity={sev_b}")
            )
        if after_arrest and sev_a != "severe":
            findings["D_severity_arrest_mismatch"].append(
                (case, p["id"], f"after is arrest but severity={sev_a}")
            )

    # Pattern E: Very low HR becoming normal without pacing or defib-capable action
    # (already covered by B, skip)

    # Pattern F: duplicate consecutive null pairs (possible over-decomposition)
    for i in range(len(pairs) - 1):
        a, b = pairs[i], pairs[i + 1]
        if (len(a["actions"]) == 0 and len(b["actions"]) == 0
            and a["after"] == b["before"]):
            findings["F_double_null_wait"].append(
                (case, f"{a['id']}+{b['id']}",
                 f"two consecutive null waits, dur={a['duration_s']}s+{b['duration_s']}s")
            )


def main():
    files = sorted(p for p in ROOT.rglob("*.json") if p.name != "schema.json")
    for f in files:
        scan_case(f)

    print(f"Scanned {len(files)} files\n")
    print("=" * 80)
    for key, items in findings.items():
        label = {
            "A_split_merge":              "Pattern A: Action+null that could MERGE",
            "B_hr_recovery_no_cpr":       "Pattern B: HR recovery w/o CPR/pacing/defib (likely missing action)",
            "C_big_change_no_cause":      "Pattern C: Large vital change in short null pair (suspicious)",
            "D_severity_arrest_mismatch": "Pattern D: Arrest vitals but severity != severe",
            "F_double_null_wait":         "Pattern F: Consecutive null waits (possible over-decomposition)",
        }.get(key, key)
        print(f"\n## {label}  [{len(items)}]")
        print("-" * 80)
        for case, pid, note in items[:50]:
            print(f"  {case:60s} {pid:10s}  {note}")
        if len(items) > 50:
            print(f"  ... and {len(items) - 50} more")

    total = sum(len(v) for v in findings.values())
    print(f"\n{'='*80}\nTotal findings: {total}")


if __name__ == "__main__":
    main()
