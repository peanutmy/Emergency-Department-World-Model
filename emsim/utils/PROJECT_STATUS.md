# EMSim Physiology Engine — Project Status

_Last updated: 2026-04-19_

**2026-04-19** — Corpus `state.drugs[]` removed (41 data-debt bugs eliminated by
feature removal). Drugs now live only in `pair.actions[]`; within-pair PD is
engine-internal on `HiddenState.active_drug_effects`. Cross-pair persistence is
a future L2 Session concern. See [DRUG_REMOVAL_REPORT.md](DRUG_REMOVAL_REPORT.md).

## 📂 Directory Layout
```
D:\wmed\emsim\
├── pdf/<category>/*.pdf              # 138 original PDFs (11 categories)
├── docs/<category>/*.docx            # 138 Word source files
├── transitions/
│   ├── schema.json                   # Global JSON Schema (current)
│   ├── EXTRACTION_GUIDE.md           # Extraction rules + default tables
│   └── <category>/*.json             # 138 transition-pair files
├── scrape_cases.py                   # Scraped emsimcases.com → case_index.json
├── download_docs.py                  # Downloaded .docx files
├── convert_pdfs.py                   # .docx → .pdf via MS Word
├── migrate_actions.py                # Schema migration (action → actions[])
└── scan_qa.py                        # QA scanner
```

## ✅ Completed

1. **138 cases fully extracted** into transition-pair JSONs, **808 pairs total**, all schema-valid.
2. **Schema finalized** after several iterations (drug-corpus field removed 2026-04-19):
   - `vitals` (6): HR, BP_sys, BP_dia, RR, O2Sat, T
   - `interventions` (18): airway, O2_device, PEEP, FiO2, vent_rate, vent_TV_ml, intubated, CPR_active, defib_last_J, pacing_active, pacing_rate, fluids_rate_ml_hr, fluid_type, warming_active, cooling_active, needle_decompression, chest_tube, pericardiocentesis
   - Drugs: appear **only in `pair.actions[]`** as bolus events (`name, dose, unit, route`); no corpus-level `state.drugs[]`
   - `mechanism.pathology`: name (snake_case) + severity (mild/moderate/severe)
3. **Pair structure**: `actions: []` (array) + `duration_s` at pair level.
4. **Sample fixes done**: Acute Respiratory Distress (p6+p7 merged), Pediatric Drowning (p8 CPR added).

## 🎬 Action Space (22 actions)

Every `action` in a pair is one of two top-level types: `intervention` (21 names) or `drug` (1 parameterized template). Drug administration is modeled as a single action with parameters (`name, dose, unit, route, age_min`), not as 54 separate discrete actions.

**21 interventions** (after `apply_O2` → `apply_nasal` merge on 2026-04-18):

| Group | Names | Notes |
|---|---|---|
| Airway & ventilation (8) | `intubate`, `apply_nasal`, `apply_NRB`, `apply_BVM`, `apply_BiPAP`, `add_PEEP`, `place_airway`, `remove_foreign_body` | `add_PEEP` carries `value` (cmH₂O); rest are name-only |
| Fluids (1) | `give_fluids` | — |
| Arrest / cardiac (4) | `start_CPR`, `stop_CPR`, `defibrillate`, `start_pacing` | — |
| Chest / thoracic (3) | `place_chest_tube`, `needle_decompress`, `pericardiocentesis` | — |
| Temperature (2) | `start_warming`, `start_cooling` | — |
| Surgical airway & misc (3) | `needle_cricothyroidotomy`, `surgical_cricothyrotomy`, `escharotomy` | rare, 1 use each |

**Drug action** — parameterized template:
```json
{ "type": "drug", "name": "epinephrine", "dose": 1, "unit": "mg", "route": "IV", "age_min": null }
```
54 unique drug names across 214 uses. Top 10: epinephrine, midazolam, norepinephrine, calcium_gluconate, propofol, atropine, amiodarone, sodium_bicarbonate, magnesium, rocuronium. Routes: IV (188), IM (11), SL (6), IV_infusion (4), PO (3), nebulized (2).

**Empty `actions: []` + `duration_s > 0`** = pure wait (249 / 808 pairs, 31%).

## ⚠️ Known Issues (from `scan_qa.py`, 103 findings)

### 🔴 High Priority (11 issues, manual fix)

**Arrest recovery without CPR** (4):
- `Endocrine/Tumour Lysis Syndrome` p7 — HR 0→70, only calcium
- `Resuscitation/Multi-Trauma Blunt VSA and Burn` p9 — HR 0→110, only cric
- `Cardiology/STEMI with Bradycardia` p4 — HR 30→200, empty actions
- `Endocrine/Hypothermia with Trauma` p7 — HR 40→180, empty actions

**Severity mismatch** (1 case, 3 records):
- `Cardiology/LVAD Pump Thrombosis` — arrest-like vitals but severity=moderate

**Suspicious Pattern C** (4):
- `Cardiology/Stable VT with ICD Firing` p5
- `Cardiology/VSA Megacode` p6, p10
- `Pediatrics/Procedural Sedation with Laryngospasm` p3

### 🟡 Medium Priority (~20, scriptable)

intubate / BVM / BiPAP + null_wait pairs that should merge like Acute Resp Distress p6+p7 was merged:
- Cardiology: VSA Megacode p12+p13
- Neurology: SAH with IICP p1+p2
- Respiratory: Airway Obstruction from FB p4+p5, COPDE p3+p4, COVID Difficult Airway p4+p5, Tracheostomy p3+p4
- Pediatrics: Acute Chest Syndrome p6+p7, Pediatric Airway Obstruction p2+p3, Pediatric Drowning p6+p7 (BVM, not the already-fixed p8)
- Communication: Critical Care 2 Myasthenic Crisis p3+p4 (BiPAP)

### 🟤 Extraction Edge Cases (from Phase 1 engine validation)

These are pairs where the extraction schema is internally consistent but
hidden an implicit co-action; Phase 1 engine validation surfaced them. Fix
by augmenting the extraction (not the engine).

- **intubate w/o add_PEEP but PEEP changes (23 pairs)**: 23/78 `intubate`
  pairs bump PEEP 0→5 without a concurrent `add_PEEP` action. Likely
  missed implicit action in extraction. Fix: add concurrent
  `add_PEEP value=5` to these pairs.
- **defibrillate w/o stop_CPR but CPR_active toggles off (8 pairs)**: 8/26
  `defibrillate`-only pairs flip `CPR_active` True→False without a concurrent
  `stop_CPR` action. Fix: add concurrent `stop_CPR`. Affected pairs:
  Cardiology/COVID-19 Out-of-Hospital Cardiac Arrest p1,
  Cardiology/COVID-19 STEMI with VF Arrest p6,
  Cardiology/Electrical Storm p6,
  Cardiology/Nightmares Case 6 Ventricular Tachycardia p6,
  Cardiology/Nightmares Case 7 Hyperkalemia p7,
  Cardiology/Ventricular Tachycardia due to ARVD p8,
  Cardiology/VSA Megacode p5,
  Resuscitation/Accidental Hypothermia p5.
- **add_PEEP name-effect mismatch (Severe Asthma p5)**: action name is
  `add_PEEP` but only `vent_rate` changes (12→10). Likely action-name typo
  in extraction. Needs case-specific review.
- **`vf_arrest` with tachy monitor HR (15 pairs)**: pathology labeled
  `vf_arrest` but `before.HR` ∈ {180, 200} — clinically this is pulseless
  VT (shockable, monitor shows the VT rate), not true VF (chaotic
  waveform, monitor reads 0). Likely extraction shortcut since both are
  shockable arrest. Fix: relabel to `pulseless_vt` or split the
  `vf_arrest` pathology. Affected: Cardiology/STEMI with Bradycardia p5-p8,
  Cardiology/VT due to ARVD p6-p8, Cardiology/VSA Megacode p3-p5,p7-p9,
  Endocrine/Hypothermia with Trauma p8-p9.
- **PEA monitor HR handled (no action needed, noted for reference)**:
  25/43 `pea_arrest` pairs have `before.HR` ∈ [20, 200]. This is
  clinically correct — PEA's defining feature is "organized electrical
  activity without pulse". Engine Phase 2 uses PEA monitor-HR pass-through
  (see ENGINE_DESIGN.md §4 HR_from_hidden). NOT a data bug; listed so
  future QA passes don't flag it.

### 🟢 Low Priority (~70, likely acceptable)

- `give_fluids + null_wait` (70% of Pattern A findings) — legitimate: fluids are persistent infusions, the flag toggle + time to observe is a reasonable split.
- Consecutive null-waits (Pattern F, 22) — mostly legitimate phase-by-phase deterioration.

## 🔧 Key Extraction Rules (reload into next session's context)

1. **Actions strictly from scenario** — no invented actions
2. **Merge vs split**: scenario modifier bundles multiple actions → one pair w/ `actions[]`; lists separately → multiple pairs
3. **`duration_s` at pair level** — shared across concurrent actions
4. **`actions: []` + `duration_s > 0`** = pure waiting (learner does nothing)
5. **Severity**: only `mild` / `moderate` / `severe`
6. **Mechanism**: only `{pathology: {name, severity}}`
7. **Drug dose default**: scenario value > max of range > guide's standard first dose
8. **FiO₂ / drug PK**: look up in `EXTRACTION_GUIDE.md` default tables
9. **State-transition default duration**: 120s when scenario doesn't specify

## 🧠 Engine Design (frozen 2026-04-18)

### Two parallel engines, same interface
Build both **Rule-based** and **Pure LLM (zero-shot)** engines. Compare with identical eval harness.

```python
class PhysiologyEngine(Protocol):
    def step(
        self,
        state: PatientState,           # full before state (schema)
        actions: list[Action],          # zero or more concurrent (22-action space)
        duration_s: float,
    ) -> PatientState: ...              # full after state
```

Pure function. No `reset()` / session state. Drug inputs come from `pair.actions[]` only; within-pair PD evolves on engine-internal `HiddenState.active_drug_effects` and is not persisted across `step()` calls (L2 Session will handle that).

### Core semantics
| Topic | Decision |
|---|---|
| Determinism | Deterministic v1 (stochasticity deferred) |
| Time stepping | Internal Δt = 30s; last step shortens to fit `duration_s` |
| Pathology | Assumed known to engine (no inference) |
| Concurrent actions | Supported (all applied at t=0) |
| Action semantics | t=0 mutates state (flip flag / set value / schedule drug effects); integration reads updated state |
| Drug tracking | Per-pair only at L1 (engine.step() re-decodes fresh HiddenState on every call). `pair.actions[]` is the sole drug input; within-pair PD lives on `h.active_drug_effects`. L2 Session will add cross-pair persistence |
| PK data | Per-class template (onset_s + effects[target, magnitude, half_life_s]) in `drug_lib.DRUG_CLASSES`; past 5 half-lives contribution ≈ 0 |

### Eval design
- Single-pair transition accuracy on all 808 pairs (no train/test split)
- **Primary**: tolerance hit-rate per vital (`HR±10, BP_sys±10, BP_dia±8, RR±3, O2Sat±3, T±0.3`, tunable)
- **Secondary**: per-vital MAE, direction correctness (↑/↓/→)
- Only compare `after.vitals` in v1 (skip `after.drugs.age_min` due to known frozen-value data gap — see `transitions/EXTRACTION_GUIDE.md` §4)
- LLM few-shot pool deferred; zero-shot first to avoid any eval-set leakage

### Code layout
```
emsim/
├── engine/
│   ├── types.py              # PatientState, Action dataclasses
│   ├── base.py               # PhysiologyEngine Protocol
│   ├── rule_based.py         # RuleBasedEngine + modifier stack
│   ├── llm_based.py          # LLMEngine + prompt templates
│   └── modifiers/            # one file per modifier
│       ├── drug_pk.py
│       ├── airway.py
│       ├── arrest.py
│       ├── fluids.py
│       └── pathology.py
├── eval/
│   ├── run_eval.py           # iterate 808 pairs, swap engines
│   └── metrics.py            # hit-rate, MAE, direction
└── transitions/              # existing (ground truth)
```

### Deferred (revisit later)
- Stochastic transitions
- Rule-based fallback for unseen pathology (currently: return unchanged + log warning)
- Multi-step trajectory prediction API
- L2 Session (cross-pair drug PD persistence, stateful hidden state)
- Agent orchestrator + patient/clinician/nurse/relative agents (just leave interface)
- LLM few-shot variants (B1 schema-only, B2 semantic kNN)

### Phase-level progress (rule-based engine)

| Phase | Status | Strict | Partial | Delivery |
|---|---|---|---|---|
| 0 — Skeleton | ✅ | 13.9% | 0.855 | rule_engine/ scaffolding, drug plumbing |
| 1 — Flag + drug plumbing | ✅ | 22.6% | 0.880 | 21 intervention flag-mutation rules |
| 2 — Rhythm infrastructure | ✅ | 22.6% | 0.880 | infer_rhythm, arrest branch, defib VF→sinus (no metric gain, as designed) |
| 3a — Decode + mapping calibration | ✅ | 22.6% | 0.880 | pathology signatures (89), residual correction, round-trip 97% |
| 3b — Encode → from_hidden | ✅ | 22.0% | 0.882 | vitals driven by hidden state; RR MAE −0.74 (intubate→vent_rate sync) |
| 4 — Pathology drift infrastructure | ✅ | 22.4% | 0.882 | 51 drift rules (42 active, 9 pinned), 8 tipping points, per-rule validation |
| 5 — Intervention effects + drift composition | ✅ | **23.3%** | — | 21 intervention hidden-state effects; drift gate unlocked for action pairs; 5/9 pinned rules unpinned; worst-20 focus transferred |
| 6 — Drug PD + multi-action composition | ✅ | **22.9%** | — | 14 class templates, 54-drug map, 14 per-drug overrides; PK curve (onset + decay); graded indication gating; drift gate configurable; per-drug/per-class diagnostic; worst-20 transferred to data-quality/arrest-emergence |
| 7 — Outlier triage + v1 freeze | ⏳ | target +2-3pp | — | review 11 high-priority issues + 14/20 data-quality worst pairs, lock canonical number |

**Phase 5 per-intervention wins** (single-action subset strict rate):
- apply_nasal: 12.5% → 59.4% (O2Sat MAE 3.06 → 1.93)
- apply_BVM: 4.5% → 36.4% (O2Sat MAE 7.41 → 5.00)
- apply_NRB: 25.0% → 41.7% (O2Sat MAE 4.62 → 3.72)
- needle_decompress: 16.7% → 33.3%

**Phase 5 findings (multi-dimensional failure bound)**:
261 of 372 remaining action_non_defib failures are multi-dimensional (HR + BP + RR all off), bounded by missing drug PD. Most intubation pairs carry concurrent propofol + rocuronium; most shock pairs carry norepi/epi; Phase 6 closes these dimensions.

**Engine design principle (from §7.0)**: each Phase's target is the infrastructure delivery criterion (worst-20 focus transfers, regression bounded, per-slice improvement), not the headline strict number. Composition across Phases is what finally moves the headline — Phase 6 is where multi-Phase composition cashes in.

## 🎯 Starting Points for Next Session

Recommended first prompt:

> "继续 emsim physiology engine 项目。138 transition JSON 在 `D:\wmed\emsim\transitions\`,QA 问题跑 `python scan_qa.py` 查。下一步:[选一个]"

Options:
- **Fix high-priority 11 issues** — arrest+no-CPR cases and LVAD severity
- **Batch-merge medium-priority ~20 intubate/BVM/BiPAP pairs** — scriptable
- **Start building stepwise engine skeleton** — use 808 pairs as test set
- **Generate `all_pairs.jsonl`** — aggregated file for batch testing
- **Build FiO₂ and drug PK default tables** — deferred "until engine time"

## 📊 Final counts

| Category | Files | Pairs |
|---|---|---|
| Cardiology | 35 | 224 |
| Resuscitation | 26 | 152 |
| Pediatrics | 16 | 98 |
| Respiratory | 15 | 81 |
| Communication | 12 | 36 |
| Neurology | 8 | 44 |
| OB-GYN | 8 | 54 |
| Endocrine | 7 | 54 |
| Trauma | 5 | 23 |
| Toxicology | 4 | 26 |
| GI | 2 | 16 |
| **Total** | **138** | **808** |

## 🧰 Useful Commands

```bash
# Validate all JSONs
cd D:/wmed/emsim && python -c "import json, jsonschema, os; s=json.load(open('transitions/schema.json')); [jsonschema.validate(json.load(open(os.path.join(r,f))), s) for r,_,fs in os.walk('transitions') for f in fs if f.endswith('.json') and f!='schema.json']; print('all valid')"

# QA scan
cd D:/wmed/emsim && python scan_qa.py

# Count pairs per category
cd D:/wmed/emsim && python -c "import json, os; [print(d, sum(len(json.load(open(f'transitions/{d}/{f}'))['pairs']) for f in os.listdir(f'transitions/{d}') if f.endswith('.json'))) for d in sorted(os.listdir('transitions')) if os.path.isdir(f'transitions/{d}')]"
```
