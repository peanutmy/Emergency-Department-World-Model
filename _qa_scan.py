import json, os, glob

ROOT = r"D:\wmed2\emsim\transitions\Cardiology"
VALID_RHYTHMS = {None, "sinus", "SVT", "VT", "VF", "asystole", "PEA", "bradycardia"}

files = sorted(glob.glob(os.path.join(ROOT, "*.json")))
print(f"Total files: {len(files)}")

total_pairs = 0
total_fields = 0
before_filled = 0
after_filled = 0
enum_dist = {"sinus":0,"SVT":0,"VT":0,"VF":0,"asystole":0,"PEA":0,"bradycardia":0,"null":0}
case_violations = []

B1_cases = {
    "Geriatric Case 2 Chronic Digoxin Toxicity.json": [("p1","before"),("p1","after"),("p2","before"),("p2","after"),("p4","before"),("p4","after"),("p5","before"),("p5","after"),("p6","before"),("p6","after")],
    "Beta Blocker Toxicity.json": [("p1","after"),("p2","before"),("p2","after"),("p3","before"),("p3","after"),("p4","before"),("p4","after")],
    "Aortic Dissection.json": [("p2","before"),("p2","after"),("p3","before"),("p3","after"),("p4","before"),("p4","after"),("p5","before"),("p5","after")],
}
B2_cases = {
    "Pregnant Cardiomyopathy.json": [("p4","before"),("p4","after"),("p5","before"),("p5","after"),("p6","before"),("p6","after")],
    "Coarctation of the Aorta.json": [("p7","before"),("p7","after"),("p11","before"),("p11","after"),("p12","before"),("p12","after")],
    "Unstable Bradycardia.json": [("p4","before"),("p4","after")],
}
B3_cases = {
    "VSA Megacode.json": [("p4","after")],
    "Aortic Dissection.json": [("p7","after")],
}

all_inconsistencies = []
pathology_rhythm = []
intervention_conflicts = []
schema_problems = []
full_pair_index = {}

def get_side(pair, side):
    s = pair.get(side, {})
    v = s.get("vitals", {})
    i = s.get("interventions", {})
    m = s.get("mechanism", {})
    p = m.get("pathology", {})
    return {
        "rhythm": m.get("rhythm", "MISSING") if "rhythm" in m else "MISSING",
        "HR": v.get("HR"),
        "BP_sys": v.get("BP_sys"),
        "BP_dia": v.get("BP_dia"),
        "CPR_active": i.get("CPR_active"),
        "pacing_active": i.get("pacing_active"),
        "pathology_name": p.get("name"),
    }

def check_vital_consistency(rhythm, HR, BPs, BPd):
    if rhythm is None or rhythm == "MISSING":
        return None
    if rhythm == "sinus":
        if HR is None: return None
        if HR < 60: return f"sinus but HR={HR} (<60)"
        if HR > 200: return f"sinus but HR={HR} (>200)"
        if BPs is not None and BPs <= 0: return f"sinus but BP_sys={BPs}"
        return None
    if rhythm == "SVT":
        if HR is not None and HR < 140: return f"SVT but HR={HR} (<140 typical)"
        if BPs is not None and BPs <= 0: return f"SVT but BP_sys={BPs}"
        return None
    if rhythm == "VT":
        if HR is not None and HR < 100: return f"VT but HR={HR} (<100)"
        return None
    if rhythm == "VF":
        if HR is not None and HR != 0: return f"VF but HR={HR} (must be 0)"
        if BPs is not None and BPs != 0: return f"VF but BP_sys={BPs}"
        if BPd is not None and BPd != 0: return f"VF but BP_dia={BPd}"
        return None
    if rhythm == "asystole":
        if HR is not None and HR != 0: return f"asystole but HR={HR}"
        if BPs is not None and BPs != 0: return f"asystole but BP_sys={BPs}"
        if BPd is not None and BPd != 0: return f"asystole but BP_dia={BPd}"
        return None
    if rhythm == "PEA":
        if BPs is not None and BPs != 0: return f"PEA but BP_sys={BPs}"
        if BPd is not None and BPd != 0: return f"PEA but BP_dia={BPd}"
        return None
    if rhythm == "bradycardia":
        if HR is not None and HR >= 60: return f"bradycardia but HR={HR} (>=60)"
        if BPs is not None and BPs <= 0: return f"bradycardia but BP_sys={BPs}"
        return None
    return None

for fp in files:
    fname = os.path.basename(fp)
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        schema_problems.append(f"{fname}: JSON parse fail: {e}")
        continue

    pairs = data.get("pairs", [])
    for pair in pairs:
        pid = pair.get("id","?")
        full_pair_index[(fname, pid)] = pair
        total_pairs += 1
        for side in ("before","after"):
            total_fields += 1
            d = get_side(pair, side)
            r = d["rhythm"]
            if r == "MISSING":
                continue
            if side == "before":
                before_filled += 1
            else:
                after_filled += 1
            if r is None:
                enum_dist["null"] += 1
                continue
            if r in VALID_RHYTHMS:
                enum_dist[r] += 1
            else:
                case_violations.append(f"{fname} {pid}.{side}: rhythm='{r}'")
                continue
            issue = check_vital_consistency(r, d["HR"], d["BP_sys"], d["BP_dia"])
            if issue:
                all_inconsistencies.append((fname, pid, side, r, d["HR"], d["BP_sys"], d["BP_dia"], issue))
            patho = (d["pathology_name"] or "").lower()
            if patho in ("vf_arrest","vfib_arrest"):
                if r not in ("VF","sinus"):
                    pathology_rhythm.append((fname, pid, side, patho, r))
            elif patho in ("pea_arrest","pea"):
                if r not in ("PEA","sinus"):
                    pathology_rhythm.append((fname, pid, side, patho, r))
            elif patho == "asystole":
                if r not in ("asystole","sinus"):
                    pathology_rhythm.append((fname, pid, side, patho, r))
            if d["CPR_active"] is True:
                if r not in ("VF","asystole","PEA","VT"):
                    intervention_conflicts.append((fname, pid, side, "CPR_active=True", r, d["HR"], d["BP_sys"]))
            if d["pacing_active"] is True:
                if r != "sinus":
                    intervention_conflicts.append((fname, pid, side, "pacing_active=True", r, d["HR"], d["BP_sys"]))

print(f"\nTotal pairs: {total_pairs}")
print(f"Total rhythm fields: {total_fields}")
print(f"Before filled: {before_filled}/{total_pairs}")
print(f"After filled: {after_filled}/{total_pairs}")
print(f"Enum dist: {enum_dist}")

print(f"\n--- A. Case violations ({len(case_violations)}) ---")
for v in case_violations:
    print("  " + v)

print("\n--- B1 (HR<60 should be brady) ---")
for fn, plist in B1_cases.items():
    for pid, side in plist:
        p = full_pair_index.get((fn, pid))
        if not p:
            print(f"  MISSING pair: {fn} {pid}")
            continue
        d = get_side(p, side)
        ok = "FIXED" if d["rhythm"] == "bradycardia" else ("OK_NULL" if d["rhythm"] is None else "WRONG")
        print(f"  [{ok}] {fn} {pid}.{side}: rhythm={d['rhythm']!r}, HR={d['HR']}")

print("\n--- B2 (HR>=60 should be sinus) ---")
for fn, plist in B2_cases.items():
    for pid, side in plist:
        p = full_pair_index.get((fn, pid))
        if not p:
            print(f"  MISSING pair: {fn} {pid}")
            continue
        d = get_side(p, side)
        ok = "FIXED" if d["rhythm"] == "sinus" else ("OK_NULL" if d["rhythm"] is None else "WRONG")
        print(f"  [{ok}] {fn} {pid}.{side}: rhythm={d['rhythm']!r}, HR={d['HR']}")

print("\n--- B3 (2 hard errors) ---")
for fn, plist in B3_cases.items():
    for pid, side in plist:
        p = full_pair_index.get((fn, pid))
        if not p:
            print(f"  MISSING pair: {fn} {pid}")
            continue
        d = get_side(p, side)
        print(f"  {fn} {pid}.{side}: rhythm={d['rhythm']!r}, HR={d['HR']}, BP={d['BP_sys']}/{d['BP_dia']}, pathology={d['pathology_name']}")

print(f"\n--- C. Global rhythm-vital inconsistencies (total {len(all_inconsistencies)}) ---")
by_r = {}
for x in all_inconsistencies:
    r = x[3]
    by_r.setdefault(r, []).append(x)
for r, lst in by_r.items():
    print(f"\n  Rhythm={r}: {len(lst)} cases")
    for x in lst[:5]:
        fname, pid, side, _r, HR, BPs, BPd, issue = x
        print(f"    {fname} {pid}.{side}: {issue}  [BP={BPs}/{BPd}]")

print(f"\n--- D. Pathology-rhythm mismatches (total {len(pathology_rhythm)}) ---")
vsa_count = sum(1 for x in pathology_rhythm if x[0] == "VSA Megacode.json")
non_vsa = [x for x in pathology_rhythm if x[0] != "VSA Megacode.json"]
print(f"  VSA Megacode (per user instruction, skipped): {vsa_count}")
print(f"  Non-VSA: {len(non_vsa)}")
for x in non_vsa[:30]:
    fname, pid, side, patho, r = x
    print(f"    {fname} {pid}.{side}: pathology={patho}, rhythm={r}")

print(f"\n--- E. Intervention conflicts ({len(intervention_conflicts)}) ---")
for x in intervention_conflicts[:30]:
    fname, pid, side, what, r, HR, BPs = x
    print(f"  {fname} {pid}.{side}: {what} but rhythm={r}, HR={HR}, BP_sys={BPs}")

print(f"\n--- Files JSON parse OK: {len(files) - len(schema_problems)}/{len(files)} ---")
for s in schema_problems:
    print("  ", s)
