import json, os, glob
ROOT = r"D:\wmed2\emsim\transitions\Cardiology"
files = sorted(glob.glob(os.path.join(ROOT, "*.json")))
nulls = []
peas = []
for fp in files:
    fname = os.path.basename(fp)
    with open(fp,"r",encoding="utf-8") as f:
        data = json.load(f)
    for pair in data.get("pairs", []):
        pid = pair.get("id","?")
        for side in ("before","after"):
            m = pair[side]["mechanism"]
            v = pair[side]["vitals"]
            i = pair[side]["interventions"]
            r = m.get("rhythm","MISSING")
            if r is None:
                nulls.append(f"{fname} {pid}.{side}: HR={v['HR']}, BP={v['BP_sys']}/{v['BP_dia']}, patho={m['pathology'].get('name')}")
            if r == "PEA":
                peas.append(f"{fname} {pid}.{side}: HR={v['HR']}, BP={v['BP_sys']}/{v['BP_dia']}, patho={m['pathology'].get('name')}, CPR={i.get('CPR_active')}")
print("NULL rhythms (8 expected):")
for n in nulls: print(" ", n)
print("\nPEA rhythms:")
for n in peas: print(" ", n)
