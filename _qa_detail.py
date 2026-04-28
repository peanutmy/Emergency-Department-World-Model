import json, os

def show(fp, pid):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    for pair in data.get("pairs", []):
        if pair.get("id") == pid:
            print(f"\n=== {os.path.basename(fp)} {pid} ===")
            print(f"comment: {pair.get('comment','')[:200]}")
            for side in ("before","after"):
                s = pair[side]
                v = s["vitals"]; m = s["mechanism"]; i = s["interventions"]
                print(f"  {side}: rhythm={m.get('rhythm')!r}, HR={v['HR']}, BP={v['BP_sys']}/{v['BP_dia']}, "
                      f"patho={m['pathology'].get('name')}, sev={m['pathology'].get('severity')}, "
                      f"CPR={i.get('CPR_active')}, pacing={i.get('pacing_active')}")
            print(f"  actions: {[a.get('name') for a in pair.get('actions',[])]}")
            return

ROOT = r"D:\wmed2\emsim\transitions\Cardiology"

show(os.path.join(ROOT,"Aortic Dissection.json"), "p2")
show(os.path.join(ROOT,"Aortic Dissection.json"), "p6")
show(os.path.join(ROOT,"Aortic Dissection.json"), "p7")
show(os.path.join(ROOT,"Coarctation of the Aorta.json"), "p11")
show(os.path.join(ROOT,"Coarctation of the Aorta.json"), "p12")
show(os.path.join(ROOT,"Unstable Bradycardia.json"), "p4")
show(os.path.join(ROOT,"VSA Megacode.json"), "p4")
show(os.path.join(ROOT,"Nightmares Case 6 Ventricular Tachycardia.json"), "p3")
show(os.path.join(ROOT,"Nightmares Case 6 Ventricular Tachycardia.json"), "p4")
show(os.path.join(ROOT,"STEMI with Bradycardia.json"), "p4")
show(os.path.join(ROOT,"STEMI with Bradycardia.json"), "p6")
show(os.path.join(ROOT,"STEMI with Bradycardia.json"), "p7")
