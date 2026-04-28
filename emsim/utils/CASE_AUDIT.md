# EMSim Case Audit Log

**Started:** 2026-04-19
**Baseline:** Post-drugs-removal (see DRUG_REMOVAL_REPORT.md)
**Goal:** Systematic review of all 138 cases / 808 pairs for extraction bugs.

---

## Audit checklist (per pair)

For each pair in each case, check all 6:

1. **Vitals consistency** — `before.vitals` / `after.vitals` match PDF state vitals (numeric); qualitative terms like "afebrile" → guide default 36.9 is OK (convention, not a bug)
2. **Actions completeness** — PDF's modifier/trigger/action mentions something not in `actions[]`? (e.g. Agitation p1 漏 `apply_NRB`)
3. **Action→state consistency** — action's implied state-field changes appear in `after`?
   - intubate → `intubated, O2_device=vent, vent_rate, vent_TV_ml, FiO2`
   - apply_NRB / apply_nasal / apply_BVM / apply_BiPAP → `O2_device, FiO2`
   - add_PEEP → `PEEP=value`
   - defibrillate → `defib_last_J`, possibly `CPR_active` change
   - start_CPR / stop_CPR → `CPR_active`
   - start_pacing → `pacing_active, pacing_rate`
   - give_fluids → `fluids_rate_ml_hr, fluid_type`
   - needle_decompress / place_chest_tube / pericardiocentesis → corresponding bool
4. **Pathology stability** — `mechanism.pathology.name` stable across pairs? severity matches vitals?
5. **Rhythm labels** — if PDF says "VF" / "pulseless VT" / "asystole" / "PEA" explicitly, does pair's pathology.name + HR pattern match?
6. **Pair structure** — merge candidates (e.g. intubate + drug that's split across 2 pair) / split candidates (2 independent transitions in 1 pair)?

## Issue tags (use in notes)

- `missing-action` — action mentioned in PDF/comment but missing from `actions[]`
- `extra-after-field` — `after` has field value with no supporting action (e.g. A1 PEEP=5)
- `severity-mismatch` — severity wrong for the vitals
- `pathology-mid-case-swap` — pathology.name changes inappropriately mid-case (e.g. Agitation p6 switches to "vt")
- `rhythm-mislabel` — VF vs pulseless VT, asystole vs PEA, etc.
- `merge-candidate` — should be merged with adjacent pair
- `split-candidate` — should be split into multiple pairs
- `scripted-rosc` — arrest recovery with no explanatory action chain
- `stop_cpr-missing` — defib pair with CPR_active toggle but no stop_CPR action
- `vitals-mismatch` — pair vitals don't match PDF scenario text
- `comment-fix` — JSON comment field is misleading/wrong (no data change needed)
- `action-name-typo` — wrong action name (e.g. add_PEEP when should be something else)
- `other` — describe in text

## Status legend (per case)

- ⏳ pending
- 🔍 in progress
- ✅ clean (no issues found)
- 🔴 has issues (listed)
- ⚠️ 0 pairs (skip — no data to audit)

## Workflow

1. Open `pdf/<Category>/<Case>.pdf` and `transitions/<Category>/<Case>.json` side-by-side
2. For each pair, run through 6-item checklist
3. Record any issues under the case's **Issues** section with tag + pair ID + brief description
4. Update status emoji in heading + `Audited: YYYY-MM-DD`
5. After finishing a category, ping me — I'll start batching fixes

## Useful commands

```bash
# Validate all JSONs
cd D:/wmed/emsim && python -c "import json, jsonschema, os; s=json.load(open('transitions/schema.json')); [jsonschema.validate(json.load(open(os.path.join(r,f), encoding='utf-8')), s) for r,_,fs in os.walk('transitions') for f in fs if f.endswith('.json') and f!='schema.json']; print('all valid')"

# Re-run eval (baseline checkpoint)
cd D:/wmed/emsim && python emsim_eval.py --transitions transitions --engine rule_engine:RuleEngine --export audit_progress.csv --worst 0
```

---

## Progress

| # | Category | Cases | Pairs | Audited | Status |
|---|---|---|---|---|---|
| 1 | Trauma | 5 | 23 | 0/5 | ⏳ |
| 2 | GI | 2 | 16 | 0/2 | ⏳ |
| 3 | Toxicology | 4 | 26 | 0/4 | ⏳ |
| 4 | Communication | 12 | 36 | 0/12 | ⏳ |
| 5 | Neurology | 8 | 44 | 0/8 | ⏳ |
| 6 | OB-GYN | 8 | 54 | 0/8 | ⏳ |
| 7 | Endocrine | 7 | 54 | 0/7 | ⏳ |
| 8 | Respiratory | 15 | 81 | 0/15 | ⏳ |
| 9 | Pediatrics | 16 | 98 | 0/16 | ⏳ |
| 10 | Resuscitation | 26 | 152 | 0/26 | ⏳ |
| 11 | Cardiology | 35 | 224 | 0/35 | ⏳ |
| | **Total** | **138** | **808** | **0/138** | |

Recommended order: smallest → largest, because Cardiology has the densest arrest/rhythm cases and it's easier to calibrate your eye on simpler cases first.

---

## Example filled entry (Agitation and Aortic Dissection, already analyzed in planning session)

#### Cardiology / Agitation and Aortic Dissection (7 pairs) 🔴 — Audited: 2026-04-19 (planning session)

Issues:
- **p1** `missing-action`: scenario描述 NRB placed during sedation; actions 只有 midazolam。Fix: 加 `{type: intervention, name: apply_NRB}`. Evidence: O2_device null→NRB, FiO2 0.21→1.0, comment 原文 "NRB was placed as part of sedation workflow".
- **p3** `comment-fix` (tentative): comment 说 "Beta blocker followed by vasoactive infusion started" 但 actions 只有 esmolol。PDF 原文需核对是否真有 vasoactive infusion。
- **p5** `extra-after-field` (A1): after.PEEP=5 but actions 只有 intubate. **Decision: B 方案** (scenario 没提 PEEP → 改 after.PEEP=0).
- **p7** `scripted-rosc` (partial): post-defib ROSC pair；after.CPR_active 仍 true 可能不对；after.pathology 切回 aortic_dissection（p6 切到 vt 算 `pathology-mid-case-swap` 的副作用，但目前 schema 无 rhythm 字段所以只能这么 encode）。

---

## Cases

### 1. Trauma (5 cases, 23 pairs)

#### Critical Care 3 – Spinal Cord Injury (6 pairs) ⏳
- Audited: —
- Issues: —

#### Gun Shot Wound (6 pairs) ⏳
- Audited: —
- Issues: —

#### Lateral Canthotomy (0 pairs) ⚠️
- Skip — no pairs to audit.

#### MVC with Tension Pneumothorax (5 pairs) ⏳
- Audited: —
- Issues: —

#### Trauma in a Hemophiliac (6 pairs) ⏳
- Audited: —
- Issues: —

---

### 2. GI (2 cases, 16 pairs)

#### Massive Upper GI Bleed (7 pairs) ⏳
- Audited: —
- Issues: —

#### Undifferentiated Abdominal Pain + Shock (9 pairs) ⏳
- Audited: —
- Issues: —

---

### 3. Toxicology (4 cases, 26 pairs)

#### ASA Overdose (6 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 11 Opioid Overdose (7 pairs) ⏳
- Audited: —
- Issues: —

#### Serotonin Syndrome (7 pairs) ⏳
- Audited: —
- Issues: —

#### Tricyclic Antidepressant Overdose (6 pairs) ⏳
- Audited: —
- Issues: —

---

### 4. Communication (12 cases, 36 pairs)

#### Alcohol and opioid use (0 pairs) ⚠️
- Skip — no pairs to audit.

#### Allyship Gendered Microaggressions (1 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Ambulatory Care (2 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Protected Intubation in the Sim Lab (2 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Respiratory Failure (4 pairs) ⏳
- Audited: —
- Issues: —

#### Critical Care 1 – Subarachnoid Hemorrhage (8 pairs) ⏳
- Audited: —
- Issues: —

#### Critical Care 2 – Myasthenic Crisis (5 pairs) ⏳
- Audited: —
- Issues: —

#### Geriatric Case 1 Delirium (0 pairs) ⚠️
- Skip — no pairs to audit.

#### Geriatric Case 4 End of Life Care (2 pairs) ⏳
- Audited: —
- Issues: —

#### Geriatric Case 6 Elder Abuse (4 pairs) ⏳
- Audited: —
- Issues: —

#### Palliative Respiratory Case (3 pairs) ⏳
- Audited: —
- Issues: —

#### Polytrauma for Team Communication (5 pairs) ⏳
- Audited: —
- Issues: —

---

### 5. Neurology (8 cases, 44 pairs)

#### Altered LOC (6 pairs) ⏳
- Audited: —
- Issues: —

#### Elderly Psychosis and Agitation (6 pairs) ⏳
- Audited: —
- Issues: —

#### Geriatric Case 5 Trauma with Head Injury (6 pairs) ⏳
- Audited: —
- Issues: —

#### Multi-trauma (Kicked off a Horse) (7 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 12 Hypertensive Encephalopathy (3 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 3 Seizure (3 pairs) ⏳
- Audited: —
- Issues: —

#### Status Epilepticus (7 pairs) ⏳
- Audited: —
- Issues: —

#### Subarachnoid Hemorrhage with Increased Intracranial Pressure (6 pairs) ⏳
- Audited: —
- Issues: —

---

### 6. OB-GYN (8 cases, 54 pairs)

#### Asthma in Pregnancy (7 pairs) ⏳
- Audited: —
- Issues: —

#### Breech Delivery + NRP (4 pairs) ⏳
- Audited: —
- Issues: —

#### Eclampsia with Apnea Secondary to Magnesium Sulfate Administration (6 pairs) ⏳
- Audited: —
- Issues: —

#### Late Post Partum Pre-eclampsia (8 pairs) ⏳
- Audited: —
- Issues: —

#### Obstetrical Trauma (8 pairs) ⏳
- Audited: —
- Issues: —

#### Postpartum Hemorrhage and NRP (9 pairs) ⏳
- Audited: —
- Issues: —

#### Resuscitative Hysterotomy (3 pairs) ⏳
- Audited: —
- Issues: —

#### Ruptured Ectopic (9 pairs) ⏳
- Audited: —
- Issues: —

---

### 7. Endocrine (7 cases, 54 pairs)

#### Adrenal Crisis (6 pairs) ⏳
- Audited: —
- Issues: —

#### CAH with adrenal crisis (5 pairs) ⏳
- Audited: —
- Issues: —

#### DKA (8 pairs) ⏳
- Audited: —
- Issues: —

#### Hyponatremic Seizure (10 pairs) ⏳
- Audited: —
- Issues: —

#### Hypothermia with Trauma (10 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric DKA (8 pairs) ⏳
- Audited: —
- Issues: —

#### Tumour Lysis Syndrome (7 pairs) ⏳
- Audited: —
- Issues: —

---

### 8. Respiratory (15 cases, 81 pairs)

#### Airway Obstruction from FB (5 pairs) ⏳
- Audited: —
- Issues: —

#### Anaphylaxis with Angioedema (9 pairs) ⏳
- Audited: —
- Issues: —

#### COPDE with Pneumothorax (6 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Difficult Airway (6 pairs) ⏳
- Audited: —
- Issues: —

#### Critical Care 4 – Post-extubation Stridor (4 pairs) ⏳
- Audited: —
- Issues: —

#### Intubation with Missing BVM (7 pairs) ⏳
- Audited: —
- Issues: —

#### Massive Hemoptysis (4 pairs) ⏳
- Audited: —
- Issues: —

#### Massive Pulmonary Embolism (4 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 2 Pneumonia (5 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 4 Pulmonary Embolism (5 pairs) ⏳
- Audited: —
- Issues: —

#### Organophosphate Poisoning (4 pairs) ⏳
- Audited: —
- Issues: —

#### PE with Bleeding (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pancreatitis with ARDS (7 pairs) ⏳
- Audited: —
- Issues: —

#### Severe Asthma Exacerbation (5 pairs) ⏳
- Audited: —
- Issues: —

#### Tracheostomy Emergency (4 pairs) ⏳
- Audited: —
- Issues: —

---

### 9. Pediatrics (16 cases, 98 pairs)

#### Acute Chest Syndrome (7 pairs) ⏳
- Audited: —
- Issues: —

#### Anaphylaxis (+- Laryngospasm) (7 pairs) ⏳
- Audited: —
- Issues: —

#### Bronchiolitis (4 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Pediatric Multisystem Inflammatory Syndrome (5 pairs) ⏳
- Audited: —
- Issues: —

#### Newborn Resuscitation (8 pairs) ⏳
- Audited: —
- Issues: —

#### Newborn Sepsis with Apneas (6 pairs) ⏳
- Audited: —
- Issues: —

#### Non-Accidental Trauma (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Airway Obstruction (4 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Asthma Exacerbation (7 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Difficult Airway (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Drowning (8 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Polytrauma (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Septic Shock (7 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Status Epilepticus (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Traumatic Brain Injury (5 pairs) ⏳
- Audited: —
- Issues: —

#### Procedural Sedation with Laryngospasm (6 pairs) ⏳
- Audited: —
- Issues: —

---

### 10. Resuscitation (26 cases, 152 pairs)

#### ASA Toxicity (10 pairs) ⏳
- Audited: —
- Issues: —

#### Accidental Hypothermia (5 pairs) ⏳
- Audited: —
- Issues: —

#### Anaphylaxis and Medication Error (6 pairs) ⏳
- Audited: —
- Issues: —

#### Armed Overdose (4 pairs) ⏳
- Audited: —
- Issues: —

#### Boating trauma (4 pairs) ⏳
- Audited: —
- Issues: —

#### Burn with COCN Toxicity (7 pairs) ⏳
- Audited: —
- Issues: —

#### Digoxin Overdose (5 pairs) ⏳
- Audited: —
- Issues: —

#### Ending a resuscitation (0 pairs) ⚠️
- Skip — no pairs to audit.

#### GSW Vascular Injury (5 pairs) ⏳
- Audited: —
- Issues: —

#### Heat related illness (3 pairs) ⏳
- Audited: —
- Issues: —

#### Inhalational Injury (2 pairs) ⏳
- Audited: —
- Issues: —

#### Intra-abdominal Sepsis (7 pairs) ⏳
- Audited: —
- Issues: —

#### Iron Overdose in a Pregnant Patient (6 pairs) ⏳
- Audited: —
- Issues: —

#### Local Anesthetic Systemic Toxicity (7 pairs) ⏳
- Audited: —
- Issues: —

#### Multi-Trauma Blunt VSA and Burn (9 pairs) ⏳
- Audited: —
- Issues: —

#### Multi-drug Overdose (12 pairs) ⏳
- Audited: —
- Issues: —

#### Multi-trauma case burn and head injury (8 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 10 Anaphylaxis (5 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 8 SepsisCholangitis (6 pairs) ⏳
- Audited: —
- Issues: —

#### Opioid Overdose with ARDS (8 pairs) ⏳
- Audited: —
- Issues: —

#### Ruptured AAA (6 pairs) ⏳
- Audited: —
- Issues: —

#### Stab Wound to the Neck with Neurogenic Shock (5 pairs) ⏳
- Audited: —
- Issues: —

#### Toxic Alcohol Ingestion (6 pairs) ⏳
- Audited: —
- Issues: —

#### Trauma Airway Management (5 pairs) ⏳
- Audited: —
- Issues: —

#### Two Patient Trauma (6 pairs) ⏳
- Audited: —
- Issues: —

#### Urologic Sepsis (5 pairs) ⏳
- Audited: —
- Issues: —

---

### 11. Cardiology (35 cases, 224 pairs)

#### Acute Respiratory Distress (6 pairs) ⏳
- Audited: —
- Issues: —

#### Agitation and Aortic Dissection (7 pairs) 🔴 — Audited: 2026-04-19 (planning session)
- **p1** `missing-action`: actions 漏 apply_NRB; evidence O2_device null→NRB + FiO2 0.21→1.0 + comment 原文 "NRB was placed as part of sedation workflow"
- **p3** `comment-fix` (tentative): comment 说 "Beta blocker followed by vasoactive infusion started" 但 actions 只有 esmolol；PDF 需核对
- **p5** `extra-after-field` (A1): after.PEEP=5 但 actions 只有 intubate. **Decision: B** (scenario 没提 PEEP) → 改 after.PEEP=0
- **p7** `scripted-rosc` (partial): post-defib ROSC；after.CPR_active 仍 true 可能不对；pathology 切回 aortic_dissection（p6 切到 vt 是 schema-limitation workaround）

#### Aortic Dissection (7 pairs) ⏳
- Audited: —
- Issues: —

#### Aortic Stenosis with A Fib and CHF (8 pairs) ⏳
- Audited: —
- Issues: —

#### Beta Blocker Toxicity (9 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 Out-of-Hospital Cardiac Arrest (3 pairs) ⏳
- Audited: —
- Issues: —

#### COVID-19 STEMI with VF Arrest (7 pairs) ⏳
- Audited: —
- Issues: —

#### Chest Pain on the Ward (7 pairs) ⏳
- Audited: —
- Issues: —

#### Coarctation of the Aorta (8 pairs) ⏳
- Audited: —
- Issues: —

#### Cocaine-induced Aortic Dissection (5 pairs) ⏳
- Audited: —
- Issues: —

#### Dysrhythmia Secondary to Hyperkalemia (6 pairs) ⏳
- Audited: —
- Issues: —

#### Electrical Storm (8 pairs) ⏳
- Audited: —
- Issues: —

#### Geriatric Case 2 Chronic Digoxin Toxicity (6 pairs) ⏳
- Audited: —
- Issues: —

#### Geriatric Case 3 Termination of Resuscitation (4 pairs) ⏳
- Audited: —
- Issues: —

#### LVAD Case (5 pairs) ⏳
- Audited: —
- Issues: —

#### LVAD Pump Thrombosis (4 pairs) ⏳
- Audited: —
- Issues: —
- Known candidate for severity change (arrest-like vitals but severity=moderate per PROJECT_STATUS 🔴 High Priority). Verify during audit.

#### Learner-Consultant Communication (3 pairs) ⏳
- Audited: —
- Issues: —

#### NIGHTMARES CASE 9 STEMI (6 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 1 Bradycardia (3 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 5 Pulmonary Edema (5 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 6 Ventricular Tachycardia (7 pairs) ⏳
- Audited: —
- Issues: —

#### Nightmares Case 7 Hyperkalemia (7 pairs) ⏳
- Audited: —
- Issues: —

#### PEA Arrest and Breaking Bad News (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric SVT (7 pairs) ⏳
- Audited: —
- Issues: —

#### Pediatric Viral Myocarditis (6 pairs) ⏳
- Audited: —
- Issues: —

#### Pregnant Cardiomyopathy (7 pairs) ⏳
- Audited: —
- Issues: —

#### STEMI with Bradycardia (9 pairs) ⏳
- Audited: —
- Issues: —
- Known candidates: p4 (HR 30→200 empty actions, `scripted-rosc`); p5-p8 vf_arrest with HR 180-200 (likely `rhythm-mislabel` pulseless VT)

#### STEMI with Cardiogenic Shock (7 pairs) ⏳
- Audited: —
- Issues: —

#### Stable VT with ICD Firing (6 pairs) ⏳
- Audited: —
- Issues: —
- Known "Pattern C" candidate: p5 (pending scan_qa.py pattern definition review)

#### TB Pericarditis (7 pairs) ⏳
- Audited: —
- Issues: —

#### Thyroid Storm (3 pairs) ⏳
- Audited: —
- Issues: —

#### Unstable Bradycardia (8 pairs) ⏳
- Audited: —
- Issues: —

#### VSA Megacode (14 pairs) ⏳
- Audited: —
- Issues: —
- Known candidates: p3-p5 + p7-p9 vf_arrest with tachy HR (likely `rhythm-mislabel`); p5 defib without stop_CPR; p6 + p10 "Pattern C"; p12+p13 `merge-candidate` (per PROJECT_STATUS 🟡)

#### Ventricular Tachycardia due to Arrhythmogenic Right Ventricular Dysplasia (ARVD) (8 pairs) ⏳
- Audited: —
- Issues: —
- Known candidates: p6-p8 vf_arrest with tachy HR (likely `rhythm-mislabel`); p8 defib without stop_CPR

#### Wide Complex Tachycardia WPW (5 pairs) ⏳
- Audited: —
- Issues: —

---

## Summary stats (to fill in at end)

| Stat | Count |
|---|---|
| Cases audited | 1 / 138 |
| Clean cases | 0 |
| Cases with issues | 1 |
| Total issues found | 4 |
| By tag — `missing-action` | 1 |
| By tag — `comment-fix` | 1 |
| By tag — `extra-after-field` (A1) | 1 |
| By tag — `scripted-rosc` | 1 |

(table auto-updatable later via script)
