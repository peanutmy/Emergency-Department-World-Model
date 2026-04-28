import json, os, glob
try:
    import jsonschema
    HAVE_JS = True
except ImportError:
    HAVE_JS = False

ROOT = r"D:\wmed2\emsim\transitions\Cardiology"
SCHEMA = r"D:\wmed2\emsim\transitions\schema.json"

with open(SCHEMA, "r", encoding="utf-8") as f:
    schema = json.load(f)

files = sorted(glob.glob(os.path.join(ROOT, "*.json")))

if HAVE_JS:
    fails = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            fails.append((os.path.basename(fp), str(e).split("\n")[0]))
        except Exception as e:
            fails.append((os.path.basename(fp), f"OTHER: {e}"))
    if fails:
        print(f"Schema FAILs: {len(fails)}")
        for f, e in fails:
            print(f"  {f}: {e}")
    else:
        print(f"Schema validation PASS (all {len(files)} files)")
else:
    print("jsonschema not available, doing manual key checks")
    # Manual checks: every state has vitals/interventions/mechanism, every mechanism has pathology
    for fp in files:
        fname = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in ("case_id","source_pdf","initial_state","pairs"):
            if k not in data:
                print(f"  {fname}: missing top key {k}")
        for pair in data.get("pairs", []):
            pid = pair.get("id","?")
            for side in ("before","after"):
                s = pair.get(side, {})
                if "vitals" not in s: print(f"  {fname} {pid}.{side}: no vitals")
                if "interventions" not in s: print(f"  {fname} {pid}.{side}: no interv")
                if "mechanism" not in s: print(f"  {fname} {pid}.{side}: no mech")
                m = s.get("mechanism", {})
                if "pathology" not in m: print(f"  {fname} {pid}.{side}: no patho")
                else:
                    p = m["pathology"]
                    if "name" not in p: print(f"  {fname} {pid}.{side}: patho no name")
                    if "severity" not in p: print(f"  {fname} {pid}.{side}: patho no sev")
                    elif p["severity"] not in ("mild","moderate","severe"):
                        print(f"  {fname} {pid}.{side}: patho.severity={p['severity']!r}")
                if "rhythm" in m:
                    r = m["rhythm"]
                    if r is not None and r not in ("sinus","SVT","VT","VF","asystole","PEA","bradycardia"):
                        print(f"  {fname} {pid}.{side}: rhythm={r!r} (invalid)")
    print("Manual check done")
