# Transition Pair Extraction Guide

This guide defines how to convert emsimcases.com scenario PDFs into transition-pair JSON files for the physiology engine test set.

## Output structure

- Write one JSON file per case at: `emsim/transitions/<category>/<case_name>.json`
- File must conform to: `emsim/transitions/schema.json`
- Reference examples:
  - `emsim/transitions/Cardiology/Acute Respiratory Distress.json`
  - `emsim/transitions/Toxicology/Opioid Overdose.json`

## Extraction principles

### 1. Strictly follow scenario text
- Only use actions/modifiers/triggers that appear in the PDF's "Scenario Progression" / "States, Modifiers, Triggers" section.
- Do NOT invent actions. Do NOT specify things the scenario left vague (e.g., "oxygen" → use generic `apply_nasal`, not `apply_NRB`).
- Only create a pair if there's a documented vital change (from state-to-state transition, or from a modifier).

### 2. State transition duration
- Use the scenario's specified time when given ("5 min", "45 sec", etc.).
- When scenario says "all actions complete → next state" without a time, default to **120 seconds (2 minutes)**.

### 3. Drug dose
- If scenario gives a specific dose → use it.
- If scenario gives a range → use the **maximum**.
- If scenario says only the drug name with no dose → use the **standard ED first dose** (see defaults table below).

### 4. Drug duration_s

Drugs appear **only in `pair.actions[]`** (as bolus events at t=0). There is no cumulative `state.drugs[]` field — the corpus-level drug-memory design was removed 2026-04-19 (see DRUG_REMOVAL_REPORT.md; data bugs around frozen `age_min`, silent drug-disappearance, and phantom drugs were eliminated by dropping the field). Within-pair pharmacodynamic evolution is handled engine-internally on `HiddenState.active_drug_effects`; cross-pair persistence is deferred to the L2 Session layer.

**For new extractions:**
- If the scenario specifies a post-dose observation time → use it as the pair's `duration_s`.
- Otherwise → default `duration_s` to the drug's onset-to-peak (seconds) from the PK table below.
- Do **not** add a `drugs` array to `before` / `after` / `initial_state`; drugs live only in `pair.actions[]`.

### 5. Severity (qualitative only)
- Three levels: `"mild" | "moderate" | "severe"`.
- Infer from clinical picture:
  - `mild` = stable vitals, minor symptoms
  - `moderate` = clearly abnormal vitals, patient uncomfortable/drowsy
  - `severe` = respiratory failure, shock, unresponsive, arrest

### 6. Mechanism is only pathology
- Schema: `{"pathology": {"name": "...", "severity": "mild|moderate|severe"}}`
- `pathology.name` = snake_case label of primary diagnosis (e.g., `pulmonary_edema`, `opioid_toxicity`, `svt`, `dka`, `tension_pneumothorax`).
- Do NOT add other fields (no fatigue, no physiology, no timers).

### 7. Pair structure & actions

Each pair has:
- `actions`: **array** of zero or more concurrent action objects
- `duration_s`: **pair-level** observation window (seconds) after all actions are taken
- `before` / `after`: full state snapshots

Action object `type`:
- `"intervention"`: needs `name`, optional `value` (no duration_s inside the action itself)
- `"drug"`: needs `name`, `dose`, `unit`, `route` (no duration_s inside the action itself)

**Rules for when to merge / split pairs**:

| Scenario situation | Representation |
|---|---|
| Single action, immediate effect | `actions: [x]`, `duration_s: 0` |
| Single action, delayed effect over time | `actions: [x]`, `duration_s: Δt` |
| Multiple actions jointly causing one transition | `actions: [x, y, ...]`, `duration_s: Δt` |
| Multiple actions that each trigger independent transitions | Multiple pairs, one per action |
| Pure time passage with NO new action taken | `actions: []`, `duration_s: Δt` |
| If/else branches (same before, different interventions) | Multiple pairs sharing the same `before` |

**Key principle**: extraction mirrors the scenario's own decomposition. If the scenario's Modifier column bundles multiple actions together (e.g., "CPR + epi → HR 100"), write them as concurrent actions in one pair. If it lists each separately with its own vital effect, split into separate pairs.

**Duration semantics**:
- `duration_s = 0`: action is purely a state change (flag toggle, equipment attach)
- `duration_s > 0`: we observe the state `duration_s` seconds after the actions were taken; delayed effects (drug peak, perfusion recovery, gas exchange re-equilibration) are captured in `after`
- For drugs with no scenario-specified observation time, use the drug's onset-to-peak in seconds (e.g., naloxone IV = 120s)
- For single intervention with no scenario time, `duration_s = 0`
- For a "wait while nothing happens" pair (no new action): `actions: []` + scenario-specified time (or 120s default for auto-progressions)

### 8. Interventions field rules
- `airway` is a **bool** exclusively for adjunct airways (OPA / NPA /
  surgical cricothyrotomy *cannula* without vent circuit / any non-ETT
  airway aid). ETT is captured by `intubated`, NOT by `airway`. Intubation
  does **not** set `airway=True` — verified against dataset: 74/78 intubate
  pairs leave `airway=False`. Only `place_airway` (OPA/NPA) and
  `needle_cricothyroidotomy` set `airway=True`; `surgical_cricothyrotomy`
  sets `intubated=True` (full ETT via surgical access).
- `O2_device` enum: `null | "nasal" | "NRB" | "BVM" | "HFNC" | "BiPAP" | "vent"`.
- `fluid_type` enum: `null | "crystalloid" | "colloid" | "blood"` (NS/LR/D5W all map to crystalloid).
- All 18 intervention fields must be present in every state (before/after/initial).
- Default intervention values: all false/null/0 unless scenario or prior action set them otherwise.

### 9. Action naming (mechanism-based)
Separate actions have distinct mechanisms (NOT merged):
- `apply_nasal` (if scenario specifies nasal prongs, or generic "oxygen" — as of 2026-04-18 the former `apply_O2` is merged into `apply_nasal`; see PROJECT_STATUS.md)
- `apply_NRB` (high-flow passive, scenario says NRB)
- `apply_BVM` (positive pressure ventilation)
- `apply_BiPAP` (as scenario specifies)
- `apply_HFNC` — legal action name but not present in current 138-case corpus; listed here for future extractions (HFNC = High-Flow Nasal Cannula).
- `add_PEEP` (value = cmH2O)
- `intubate` (creates ETT + vent)
- `place_airway` (merged: OPA or NPA — scenario says "airway adjunct" or "OPA or NPA")
- `start_CPR`, `stop_CPR`
- `defibrillate` (value = joules)
- `start_pacing` (value = rate)
- `give_fluids` (value = ml/hr)
- `needle_decompress`, `place_chest_tube`, `pericardiocentesis`
- `start_warming`, `start_cooling`

## FiO2 defaults by device

| O2_device | FiO2 |
|---|---|
| null (RA) | 0.21 |
| nasal | 0.3 |
| NRB | 1.0 |
| BVM | 1.0 |
| HFNC | 0.6 |
| BiPAP | 0.5 |
| vent | 1.0 (initial) |

## Drug PK defaults (IV unless noted)

Fill `duration_s` from onset-to-peak (seconds) when scenario doesn't specify a post-dose observation time:

| Drug | Onset-to-peak (min) | Default first dose |
|---|---|---|
| naloxone | 2 | 0.4 mg |
| epinephrine (ACLS) | 1 | 1 mg |
| epinephrine (anaphylaxis IM) | 5 | 0.3 mg |
| amiodarone | 5 | 300 mg |
| atropine | 2 | 0.5 mg |
| adenosine | 0.3 (20 s) | 6 mg |
| fentanyl | 5 | 100 mcg |
| midazolam | 3 | 2 mg |
| ketamine | 1 | 100 mg (1.5 mg/kg × 80 kg; round) |
| succinylcholine | 1 | 120 mg (1.5 mg/kg × 80 kg) |
| rocuronium | 2 | 80 mg (1 mg/kg × 80 kg) |
| propofol | 1 | 150 mg (2 mg/kg × 80 kg) |
| calcium_gluconate | 2 | 1 g |
| dextrose_50 | 5 | 25 g |
| glucagon | 10 | 5 mg |
| insulin_regular (DKA) | 10 | 8 U (0.1 U/kg × 80 kg; bolus) |
| nitroglycerin_SL | 3 | 0.4 mg |
| furosemide | 30 | 40 mg |
| magnesium | 10 | 2 g |
| hydrocortisone | 15 | 100 mg |
| flumazenil | 2 | 0.2 mg |
| naltrexone | 15 | 50 mg |
| sodium_bicarbonate | 2 | 50 mEq (1 amp) |
| tPA (stroke) | 30 | 72 mg (0.9 mg/kg × 80 kg; max 90) |
| TXA | 5 | 1 g |
| ondansetron | 15 | 4 mg |
| haloperidol | 15 | 5 mg |

## Common pathology names (snake_case)

- Cardiac: `stemi`, `nstemi`, `svt`, `vt`, `vf_arrest`, `asystole`, `pea_arrest`, `bradycardia`, `aortic_dissection`, `pericarditis`, `tamponade`, `chf_exacerbation`, `pulmonary_edema`, `cardiogenic_shock`
- Respiratory: `asthma_exacerbation`, `copd_exacerbation`, `pulmonary_embolism`, `pneumonia`, `pneumothorax`, `tension_pneumothorax`, `ards`, `respiratory_failure`, `anaphylaxis`, `angioedema`, `airway_obstruction`
- Tox: `opioid_toxicity`, `benzo_toxicity`, `tca_overdose`, `asa_toxicity`, `toxic_alcohol`, `beta_blocker_toxicity`, `ccb_toxicity`, `digoxin_toxicity`, `serotonin_syndrome`, `organophosphate_poisoning`, `iron_overdose`, `cyanide_toxicity`
- Endo/Metabolic: `dka`, `thyroid_storm`, `adrenal_crisis`, `hyponatremia`, `hyperkalemia`, `hypokalemia`, `hypoglycemia`, `tumor_lysis`, `hyperthermia`, `hypothermia`
- Neuro: `status_epilepticus`, `ischemic_stroke`, `subarachnoid_hemorrhage`, `traumatic_brain_injury`, `delirium`
- Trauma: `hemorrhagic_shock`, `polytrauma`, `burn`, `inhalation_injury`, `spinal_cord_injury`, `blunt_abdominal_trauma`
- Sepsis/infection: `septic_shock`, `sepsis`, `cholangitis`, `meningitis`
- OB/GYN: `eclampsia`, `preeclampsia`, `postpartum_hemorrhage`, `ectopic_pregnancy`, `amniotic_embolism`
- GI: `upper_gi_bleed`, `pancreatitis`, `bowel_obstruction`

Pick the closest match; invent new snake_case if none fits.

## Workflow per PDF

1. Read the PDF (use the Read tool on the PDF path).
2. Find the "Scenario Progression" / "States, Modifiers, Triggers" section.
3. Note the baseline vitals and pathology.
4. For each state transition (triggered by action, modifier, or time), construct a pair:
   - `before` = previous state's vitals + cumulative interventions
   - `actions` = the triggers (zero or more intervention / drug actions; empty for pure-wait pairs)
   - `after` = new vitals + updated interventions
5. Include branch pairs (parallel alternatives from the same before state) when scenario's modifiers describe them.
6. Include inaction pairs (`type: null`) when scenario describes what happens if the learner doesn't intervene.
7. Validate the JSON against schema before finalizing.

## Validation command

```bash
cd D:/wmed/emsim && python -c "
import json, jsonschema
schema = json.load(open('transitions/schema.json'))
data = json.load(open('transitions/<category>/<case>.json'))
jsonschema.validate(data, schema)
print('valid,', len(data['pairs']), 'pairs')
"
```

## Output quality checklist

Before finalizing a case, check:
- [ ] Every pair's `before.vitals` == the previous pair's `after.vitals` (linear path) OR explicitly a branch
- [ ] All 18 intervention fields present in every state
- [ ] Severity only uses mild/moderate/severe
- [ ] mechanism has exactly one key: `pathology`
- [ ] No invented action names (only used what scenario documented)
- [ ] JSON validates against schema
