#!/usr/bin/env python
"""One-off: fill chief_complaint / symptoms / baseline_vitals / turnaround_turns
into one scenario per transitions subfolder, from the PDF ground truth.

Run once:  python fill_scenarios.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANS = ROOT / "transitions"

# turnaround_turns by test_category (sim timing knob; anchored to the reference:
# ecg=2, imaging=3). Bedside/point-of-care = 1; standard labs = 2.
TURNAROUND = {
    "ecg": 2, "imaging": 3,
    "blood_gas": 1, "ultrasound": 1, "echo": 1,
    "lab": 2, "hematology": 2, "cardiac_test": 2,
    "coagulation": 2, "toxicology": 2, "endocrine_test": 2,
}
TURNAROUND_DEFAULT = 2

# PDF-grounded data (initial/presenting vitals; values only where the PDF states them).
DATA = {
    "Communication/Critical Care 1 – Subarachnoid Hemorrhage": {
        "chief_complaint": "Headache with decreased level of consciousness",
        "symptoms": ["Severe headache", "Confusion", "Decreased level of consciousness",
                     "Sluggishly reactive pupils", "Decerebrate posturing to pain"],
        "baseline_vitals": {"HR": 110, "BP_sys": 170, "BP_dia": 95, "RR": 12, "O2Sat": 92, "T": 37.4},
    },
    "Endocrine/DKA": {
        "chief_complaint": "Altered level of consciousness",
        "symptoms": ["Fatigue", "Vomiting", "Mild abdominal pain", "Confusion", "Lethargy",
                     "Dry mucous membranes", "Kussmaul (deep) breathing", "Polyuria",
                     "Acetone breath", "Altered mental status"],
        "baseline_vitals": {"HR": 140, "BP_sys": 90, "BP_dia": 50, "RR": 34, "O2Sat": 99, "T": 36.5},
    },
    "GI/Massive Upper GI Bleed": {
        "chief_complaint": "Massive hematemesis",
        "symptoms": ["Hematemesis", "Nausea", "Chest discomfort", "Confusion", "Drowsiness",
                     "Clinical intoxication (EtOH)", "Blood in oropharynx", "Jaundice",
                     "Cool skin", "Cirrhotic habitus"],
        "baseline_vitals": {"HR": 115, "BP_sys": 105, "BP_dia": 60, "RR": 24, "O2Sat": 96, "T": 37.8},
    },
    "Neurology/Status Epilepticus": {
        "chief_complaint": "Seizure",
        "symptoms": ["Active generalized tonic-clonic seizure", "Prolonged seizure",
                     "Non-reactive pupils", "Unresponsiveness"],
        "baseline_vitals": {"HR": 130, "BP_sys": 130, "BP_dia": 80, "RR": 12, "O2Sat": 96, "T": 36.4},
    },
    "OB-GYN/Ruptured Ectopic": {
        "chief_complaint": "Syncope",
        "symptoms": ["Dizziness", "Syncope", "Vomiting", "Vague abdominal pain", "Weakness",
                     "Pallor", "Diffuse abdominal tenderness"],
        "baseline_vitals": {"HR": 100, "BP_sys": 90, "BP_dia": 60, "RR": 12, "O2Sat": 98, "T": 36.5},
    },
    "Pediatrics/Pediatric Asthma Exacerbation": {
        "chief_complaint": "Shortness of breath",
        "symptoms": ["Cough", "Wheeze", "Worsening shortness of breath",
                     "Increased work of breathing", "Decreased air entry", "Chest tightness",
                     "Tracheal tug", "Intercostal indrawing", "Rhinorrhea", "Appears frightened"],
        "baseline_vitals": {"HR": 160, "BP_sys": 80, "BP_dia": 50, "RR": 40, "O2Sat": 99, "T": 37.8},
    },
    "Respiratory/COPDE with Pneumothorax": {
        "chief_complaint": "Shortness of breath",
        "symptoms": ["Worsening cough", "Increased sputum", "Dyspnea",
                     "Severe respiratory distress", "Agitation", "Gasping for air",
                     "Diffuse wheeze", "Hypoxia", "Decreased responsiveness"],
        "baseline_vitals": {"HR": 120, "BP_sys": 160, "BP_dia": 95, "RR": 30, "O2Sat": 80, "T": 37.0},
    },
    "Resuscitation/Intra-abdominal Sepsis": {
        "chief_complaint": "Abdominal pain",
        "symptoms": ["Abdominal pain", "Fever", "Palpitations", "Nausea and vomiting",
                     "Right upper quadrant tenderness with rebound and guarding",
                     "Cool, clammy extremities", "Jaundice"],
        "baseline_vitals": {"HR": 120, "BP_sys": 85, "BP_dia": 55, "RR": 20, "O2Sat": 95, "T": 39.5},
    },
    "Toxicology/Tricyclic Antidepressant Overdose": {
        "chief_complaint": "Altered mental status",
        "symptoms": ["Altered mental status", "Brief seizure activity", "Somnolence",
                     "Unresponsiveness", "Dilated, poorly reactive pupils", "Tachycardia",
                     "Absent bowel sounds", "Urinary retention", "Flushed skin", "Fever"],
        "baseline_vitals": {"HR": 122, "BP_sys": 109, "BP_dia": 71, "RR": 14, "O2Sat": 92, "T": 38.1},
    },
}

KEY_ORDER = ["demographics", "chief_complaint", "symptoms", "history",
             "home_medications", "allergies", "baseline_vitals",
             "baseline_features", "supporting_findings"]


def reorder(cc: dict) -> dict:
    out = {k: cc[k] for k in KEY_ORDER if k in cc}
    for k, v in cc.items():  # keep any extra keys at the end
        if k not in out:
            out[k] = v
    return out


def apply(path: Path, entry: dict) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cc = data["case_context"]
    cc.pop("Symptoms", None)  # drop the non-loading capitalised key if present
    cc["chief_complaint"] = entry["chief_complaint"]
    cc["symptoms"] = entry["symptoms"]
    cc["baseline_vitals"] = entry["baseline_vitals"]
    findings = cc.get("supporting_findings") or []
    for f in findings:
        f["turnaround_turns"] = TURNAROUND.get(f.get("test_category"), TURNAROUND_DEFAULT)
    data["case_context"] = reorder(cc)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ta = [(f.get("test_name") or f.get("test_code"), f["turnaround_turns"]) for f in findings]
    return f"vitals={entry['baseline_vitals']}  findings_turnaround={ta}"


print("=== Filled scenarios ===")
for rel, entry in DATA.items():
    p = TRANS / f"{rel}.json"
    print(f"\n[{rel}]")
    print("  ", apply(p, entry))

# Fix the reference file's capitalised "Symptoms" -> lowercase "symptoms" so it loads.
ref = TRANS / "Cardiology/Acute Respiratory Distress.json"
rd = json.loads(ref.read_text(encoding="utf-8"))
rcc = rd["case_context"]
if "Symptoms" in rcc and "symptoms" not in rcc:
    rcc["symptoms"] = rcc.pop("Symptoms")
    rd["case_context"] = reorder(rcc)
    ref.write_text(json.dumps(rd, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n[reference] fixed Acute Respiratory Distress: 'Symptoms' -> 'symptoms'")
