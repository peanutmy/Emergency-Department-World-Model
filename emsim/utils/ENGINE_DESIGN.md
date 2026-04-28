# EMSim Physiology Engine — Design & Build Plan

_v1 · 2026-04-18_

---

## 0. North Star

**Vitals accuracy is the primary metric.** 6 observable vitals per pair, measured
against the 808 authored pairs under the tolerance bands defined in
`emsim_eval.py`. Every design choice gets evaluated on one question:

> Does this change reduce per-vital MAE and increase direction-match rate?

Flag / mechanism / drug matching are secondary — they're scaffolding that helps
vitals be correct, but they don't earn points on their own.

## 1. Architectural Principle

Three-layer separation:

```
        ┌──────────────────────────────────────────────┐
        │  Schema state dict (vitals + flags)          │ ← I/O contract
        └────────┬─────────────────────────────────▲───┘
                 │ decode                   encode │
                 ▼                                 │
        ┌──────────────────────────┐   ┌──────────────────┐
        │  HiddenState (12 vars)   │──▶│ Vitals mapping   │
        │  rhythm, CO, SVR, PaO2…  │   │ (hidden → 6 vit) │
        └──────────┬───────────────┘   └──────────────────┘
                   │ mutated by
                   ▼
        ┌──────────────────────────────────────────────┐
        │ Rules (pathology drift · interventions · drug PD) │
        └──────────────────────────────────────────────┘
```

Rules **never** touch vitals directly. They mutate hidden state; vitals are a
pure function of hidden state. This is what makes rules generalize — a new
pathology only has to specify its effect on hidden vars, and the downstream
vitals changes emerge automatically.

## 2. File Layout

Drop inside `D:\wmed\emsim\rule_engine\`:

```
rule_engine/
├── __init__.py           # exports RuleEngine, EngineSession
├── engine.py             # L1: main step() orchestrator
├── session.py            # L2: EngineSession (stateful agent-facing wrapper)
├── orchestrator.py       # L3: OrchestratorProtocol (interface only, v1 stub)
├── events.py             # Event dataclasses (unused in v1, importable)
├── hidden_state.py       # @dataclass HiddenState + DrugEffect
├── mapping.py            # vitals_from_hidden(), hidden_from_state() seed
├── pathology_lib.py      # pathology_effect() + init_signature() registries
├── intervention_lib.py   # intervention_effect() registry
├── drug_lib.py           # drug classes + PD templates
└── io.py                 # schema state <-> HiddenState conversion
```

Plug into existing harness:
```bash
python emsim_eval.py --transitions transitions --engine rule_engine:RuleEngine
```

### 2.1 Registry name-pinning

The rule libraries expose a small number of module-level dicts that decorators
populate at import time. These names are part of the engine's internal contract
— stable across phases, referenced by decode/encode and the integrator loop:

| Registry / constant          | Module              | Role |
|------------------------------|---------------------|------|
| `PATHOLOGY_RULES`            | `pathology_lib.py`  | name → drift rule `fn(h, severity, dt_s)` |
| `PATHOLOGY_INIT_SIGNATURES`  | `pathology_lib.py`  | name → init signature `fn(severity) -> dict` |
| `DEFAULT_HEALTHY`            | `pathology_lib.py`  | baseline hidden-state dict for unspecified vars |
| `INTERVENTION_RULES`         | `intervention_lib.py` | name → jump rule `fn(h, flags, params)` |
| `DRUG_CLASSES`               | `drug_lib.py`       | class_name → PD template dict |
| `DRUG_TO_CLASS`              | `drug_lib.py`       | drug_name → class_name |
| `DRUG_OVERRIDES`             | `drug_lib.py`       | per-drug extra effects (ketamine etc.) |
| `DRUG_STANDARD_DOSE`         | `drug_lib.py`       | class-standard dose for linear scaling |

## 3. Hidden State (12 vars — frozen for v1)

All indices are dimensionless multipliers around 1.0 unless stated. Ranges are
"reasonable" clips, not hard physics.

| # | Var | Unit / Range | What it drives |
|---|-----|--------------|----------------|
| 1 | `rhythm` | enum: sinus, SVT, VT, VF, asystole, PEA, bradycardia | HR floor/shape; arrest ⇒ no pulse |
| 2 | `CO_index` | physiology [0, 2.5] / residual-fit [0.05, 4.0], baseline 1.0 | MAP (multiplicative) |
| 3 | `SVR_index` | [0.3, 2.5], baseline 1.0 | MAP (multiplicative); also dia/sys ratio |
| 4 | `preload_index` | [0, 1.8], baseline 1.0 | CO (via Frank-Starling proxy) |
| 5 | `chronotropic_drive` | [-2, +3] (SD units), baseline 0 | HR offset within a rhythm |
| 6 | `PaO2_effective` | mmHg, [0, 600] | O2Sat via dissociation curve (min=0 to allow arrest reverse-mapping) |
| 7 | `shunt_fraction` | [0, 0.8] | FiO2 responsiveness of PaO2 |
| 8 | `compliance_index` | [0.2, 1.2], baseline 1.0 | PEEP/vent efficacy on shunt |
| 9 | `ventilatory_drive` | [0, 2.5], baseline 1.0 | spontaneous RR (overridden if intubated) |
| 10 | `core_temp_trend` | °C/min, default **0 in Phase 3 signatures** (non-zero trends are Phase 4 drift territory) | T drift direction |
| 11 | `consciousness` | [0, 1], baseline 1 | airway protection, spontaneous RR gating |
| 12 | `active_drug_effects` | list[DrugEffect] | within-pair PD contributions; fresh per `step()` at the L1 stateless boundary (L2 Session, §6.5, will persist this across pairs) |

`DrugEffect` = `{target_var, magnitude, t_start_s, half_life_s}`. Integrator
decays magnitude by half every `half_life_s` and applies to its target on each tick.

### 3.1 Var-semantics notes (pinned)

- **`rhythm`** is stored as a **string** (matching the JSON-schema style),
  not a Python `Enum`. Legal values:
  `"sinus" | "SVT" | "VT" | "VF" | "asystole" | "PEA" | "bradycardia"`.
  The frozenset `ARREST_RHYTHMS = {"asystole", "VF", "PEA"}` in
  `hidden_state.py` is the canonical check for no-pulse states — always
  prefer `h.rhythm in ARREST_RHYTHMS` over open-coding the tuple.
- **`consciousness`** has baseline **1.0 (fully conscious)**, range `[0, 1]`.
  Unlike the 1.0-centered multiplier vars (`CO_index`, `SVR_index`,
  `preload_index`, `compliance_index`, `ventilatory_drive`), consciousness
  is a **clamped proportion** (0 = unresponsive, 1 = alert). Rules should
  clamp writes, not extrapolate past the unit interval.
- **`DEFAULT_HEALTHY`** lives in `pathology_lib.py` (owned by the pathology
  domain); `io.py` and `engine.py` import it. The `HiddenState` dataclass
  field defaults currently duplicate these values for convenience; this is
  tracked as tech debt (see §13). New hidden-state vars added going forward
  should **not** introduce further duplication — declare the baseline in
  `DEFAULT_HEALTHY` only, and let `decode_state()` overlay it onto
  `HiddenState` at construction time.

## 3.5. Hidden State Initialization

**Problem.** Engine only receives `before` — 6 vitals + flags + pathology name/severity.
We have 12 hidden vars to populate. 6 equations, 12 unknowns → underdetermined.
We resolve this with **pathology-conditioned initialization** (two-step).

### Step 1 — Pathology signature

Every pathology registers an `init_signature(severity)` in `pathology_lib.py` that
returns the hidden-state "shape" typical of that pathology. Severity is a scalar
(mild=0.3, moderate=0.6, severe=1.0) that interpolates the magnitude of the shape.

```python
@pathology_init("pulmonary_edema")
def pe_init(severity: float) -> dict:
    return {
        "rhythm":             "sinus",
        "shunt_fraction":     0.15 + 0.35 * severity,   # severe → shunt ≈ 0.5
        "compliance_index":   1.0  - 0.5  * severity,
        "preload_index":      1.0  + 0.3  * severity,   # volume overload
        "CO_index":           1.0  - 0.2  * severity,
        "ventilatory_drive":  1.0  + 0.8  * severity,   # tachypnea
        "chronotropic_drive": 0.5  * severity,
        # unspecified vars → default healthy baseline
    }
```

Vars not listed default to `DEFAULT_HEALTHY`. This way each pathology author only
has to declare the physiologically relevant dimensions.

### Step 2 — Residual correction

The signature gets us 80% there; observed vitals tell us the last 20%.
For each vital, if `predicted_vital(h) ≠ observed_vital` by more than tolerance,
nudge **one designated hidden var** toward closing the gap. Each vital has
a single designated var so we don't create ambiguity:

| Vital | Designated hidden var | Rationale |
|-------|-----------------------|-----------|
| HR    | `chronotropic_drive`  | rhythm already fixed by signature |
| BP_sys + BP_dia | `SVR_index` + `CO_index` (dual-var analytic solve) | single-var fails ~20 pairs with wide/narrow PP; dual-var analytic invert from (BP_sys, BP_dia) → (CO_eff, SVR) is needed to hit ≥95% round-trip. See `io.py:_residual_correct`. |
| RR    | `ventilatory_drive`   | |
| O2Sat | `PaO2_effective`      | |
| T     | (no correction, direct copy) | T is the only vital we store raw |

**Pulseless VT / SVT special case**: some pairs label rhythm as VT/SVT (non-arrest) but observe `BP_sys=0, BP_dia=0` with `HR>0` — this is clinically pulseless tachyarrhythmia. Residual correction handles this by forcing `CO_index=0` when both BPs are 0, keeping the rhythm label for downstream logic (defibrillation pathway) but zeroing the BP output. See `io.py:_residual_correct`.

Note: T has no residual-correction row because it's the only vital stored
raw (not derived from an index). See §4 `T_from_hidden` — during forward
integration T evolves as `T_before + core_temp_trend * dt_min`, so "direct
copy" in this table refers specifically to hidden-state initialization,
not runtime evolution.

Rhythm is categorical — **never adjusted by residual correction**; it stays
whatever the signature assigns. If observed HR contradicts rhythm (e.g. HR 180
with signature rhythm=sinus), the signature is wrong — fix the signature, not
the initial state.

### Fallback

If `before.mechanism.pathology.name` isn't registered in the init library:
- Log a warning
- Fall back to `DEFAULT_HEALTHY` baseline
- Apply residual correction on all vitals with a designated hidden var
  (i.e. HR, BP_sys, RR, O2Sat — T is excluded as noted in Step 2)

This keeps the engine running but flags the gap. Registering a new init_signature
is the fix.

## 4. Vitals Mapping (the most load-bearing module)

6 functions, one per vital. Documented starting formulas — all coefficients are
tunable and will be calibrated against the 808 pairs.

```python
def HR_from_hidden(h, before_HR) -> float | None:
    # IMPORTANT: monitor HR ≠ perfusion HR. Dataset evidence (Phase 2):
    # - asystole: monitor flatline → HR=0
    # - VF: chaotic waveform → dataset consistently records HR=0
    # - PEA: "pulseless electrical activity" → monitor still shows organized
    #   electrical rate (20–200 bpm in 25/43 pea_arrest pairs). Clinically
    #   the defining feature of PEA is monitor HR present but no palpable
    #   pulse. Engine must record monitor HR, so for PEA we pass-through.
    if h.rhythm in ("asystole", "VF"):  return 0
    if h.rhythm == "PEA":               return before_HR  # monitor rate preserved
    if h.rhythm == "VT":                return clip(180 + 10 * h.chronotropic_drive, 150, 220)
    if h.rhythm == "SVT":               return clip(170 + 10 * h.chronotropic_drive, 140, 220)
    if h.rhythm == "bradycardia":       return clip(45  + 8  * h.chronotropic_drive, 20, 60)
    # sinus — Phase 3 will calibrate; Phase 2 returns None → pass-through
    return None   # Phase 3: clip(75 + 15 * h.chronotropic_drive, 40, 180)

def BP_from_hidden(h, before_BP_sys, before_BP_dia) -> tuple[float, float] | tuple[None, None]:
    # asystole/VF: no ventricular output → monitor BP=0 (consistent across dataset)
    # PEA: electrical activity without effective mechanical contraction; dataset
    #      shows BP=0 on all PEA arrest pairs (cuff reads 0 without perfusion).
    #      So BP zeroing applies to all three arrest rhythms, unlike HR.
    if h.rhythm in ("asystole", "VF", "PEA"): return (0, 0)
    # Phase 3 will calibrate the Ohm's law analog below; Phase 2 pass-through.
    return (None, None)
    # Phase 3 implementation:
    #   CO_eff = h.CO_index * frank_starling(h.preload_index)
    #   MAP    = 90 * CO_eff * h.SVR_index
    #   pulse  = 40 * CO_eff                  # simplified from 40*CO_eff/SVR
    #   return (MAP + pulse/2, MAP - pulse/2)
    # NOTE (Phase 3a): PP originally coupled to SVR (40*CO_eff/SVR_index) to
    # model narrow PP under vasoconstriction. But that couples CO_eff and SVR
    # in a way that defeats the dual-var analytic invert from (BP_sys, BP_dia).
    # Dropped to PP = 40*CO_eff (physiologically equivalent at baseline
    # CO=SVR=1). SVR-driven PP narrowing can return as a Phase 5+ refinement
    # if needed; Phase 3a calibration doesn't require it.
    #
    # KNOWN LIMITATION (Phase 3b): the decoupled form cannot represent
    # `sys == dia > 0` (degenerate narrow-PP, e.g. LVAD continuous flow).
    # PP=0 requires CO_eff=0 which also zeros MAP. 2 LVAD pairs hit this;
    # `encode_state` uses a narrow-PP safety net (pass-through `before.BP`
    # when `0 ≤ sys − dia < 5` and `sys > 0`). Safety net is removable once
    # BP gets an independent PP var (or switches to iterative residual fit).

def RR_from_hidden(h, flags) -> float:
    # Intubated override gate: requires BOTH intubated=True AND vent_rate
    # actually set. Pre-existing trach / BVM-no-set-rate states have
    # intubated=True but vent_rate=None (5 pairs in dataset) — they need
    # spontaneous fall-through to ventilatory_drive rather than synthetic
    # default 12. `decode_state` residual gate mirrors this condition.
    if flags["intubated"] and flags.get("vent_rate") is not None:
        return flags["vent_rate"]
    # NOTE (Phase 3a): consciousness multiplier removed from RR formula.
    # Original "14 * ventilatory_drive * consciousness" made round-trip
    # impossible for BVM/rescue-breathing arrest pairs (consciousness=0 but
    # observed RR=12). ventilatory_drive alone carries the obtundation signal:
    # Phase 4 pathology drift pulls ventilatory_drive down directly during
    # decompensation. Consciousness still gates airway protection behavior
    # elsewhere, just not RR output.
    return clip(14 * h.ventilatory_drive, 0, 50)

def O2Sat_from_hidden(h, flags, before_O2Sat) -> float | None:
    # NOTE: do NOT zero O2Sat on arrest rhythm. Dataset evidence (Phase 2
    # validation, 2026-04-18): 21 before.HR==0 pairs keep before.O2Sat>0 and
    # carry it into after.O2Sat>0 — this is the pulse-ox "last-valid-hold"
    # artifact, not a true 0 reading. Zeroing hurts 21 pairs, helps 1. The
    # arrest branch for HR and BP is still correct (monitor shows 0 HR / 0 BP
    # on PEA/VF/asystole), but O2Sat is instrument-held.
    # Return None for pass-through until Phase 3 calibrates PaO2_effective.
    return None   # Phase 3 will return hemoglobin_curve(h.PaO2_effective)

def T_from_hidden(h, T_before, dt_s) -> float:
    return T_before + h.core_temp_trend * (dt_s / 60)
    # KNOWN LIMITATION (Phase 3b): pathology init_signatures that set
    # non-zero `core_temp_trend` (e.g. hypothermia, hyperthermia) accumulate
    # drift over long-duration pairs (900–1800s) and collide with the eval
    # T direction-match dead-zone (0.1 °C). `encode_state` uses a tiny-drift
    # safety net (pass-through `T_before` when predicted drift < 0.2 °C).
    # Phase 4 will replace signature-set trends with scenario-paced drift
    # rules, at which point the safety net is removable.
```

`hemoglobin_curve` is a cached lookup table (PaO2 → O2Sat). Key anchor points:
PaO2 40 → 75%, 60 → 90%, 80 → 95%, 100 → 98%, 150 → 100%.

`mapping.py` also exports utility helpers `clip(value, lo, hi)` and
`lerp(a, b, weight)` used throughout the intervention rules (see §5
`intubate` example).

**Calibration check (Phase 3):** for every pair's `before` state, the mapping
applied to the decoded HiddenState should reproduce `before.vitals` within
tolerance. If not, the decoder or mapping formula is wrong — fix before adding
any drift rules.

### 4.1 Coefficient provenance and calibration policy

The formulas above mix three levels of confidence. Keeping this distinction
visible in comments matters when debugging:

| Level | Examples | Treatment |
|-------|----------|-----------|
| **Real physiology** | `MAP = CO × SVR` (Ohm's law analog), hemoglobin curve anchors (40/60/80/100 mmHg), arrest rhythms ⇒ no pulse | Structure fixed, not tuned |
| **Plausible functional form** | `MAP = 90 × CO × SVR` (the `90` is the baseline), sinus HR = `75 + 15 × drive`, RR = `14 × drive × consciousness` | Structure kept, coefficients tunable |
| **Author-constructed proxy** | Pulse pressure = `40 × CO / SVR`, `ventilatory_drive` as a unitless scalar | Functional form itself may change if calibration fails |

Every coefficient in `mapping.py` is a named constant, not a magic number:

```python
HR_SINUS_BASELINE = 75       # bpm at drive=0
HR_SINUS_GAIN     = 15       # bpm per unit drive
MAP_BASELINE      = 90       # mmHg at CO=SVR=1.0
PP_BASELINE       = 40       # mmHg at CO=SVR=1.0  [low-confidence form]
# ...
```

**Calibration in Phase 3** fits these constants against the 808 `before` states,
minimizing round-trip reproduction error. If a low-confidence form can't be
calibrated to within tolerance on ≥95% of pairs, the form gets replaced — not
the coefficients tightened indefinitely.

**Medical review encouraged.** All `mapping.py` and `intervention_lib.py` direction
signs (not magnitudes) should get a 30-min sanity check from an EM/ICU clinician
before Phase 4 starts. Catches wrong-direction bugs cheap.

## 5. Rule Libraries (3 modules, uniform signatures)

### Pathology (drift over time)

Drift rule coefficients are Phase-4-calibrated, not physiology-derived.
All numeric literals in sample rules below (e.g. `0.002`, `0.005`) are
author-initial starting points. Phase 4 calibration fits them against the
249 pure-wait pairs to match scenario-paced drift. Rule authors should tag
coefficients with the same confidence levels as §4.1 (real physiology /
plausible form / author-constructed) in code comments.

```python
# pathology_lib.py
PATHOLOGY_RULES = {}  # name → rule function

def pathology_rule(name):  # decorator
    def register(fn): PATHOLOGY_RULES[name] = fn; return fn
    return register

@pathology_rule("pulmonary_edema")
def pe_drift(h: HiddenState, severity: float, dt_s: float) -> None:
    """severity ∈ {0.3, 0.6, 1.0} for mild/moderate/severe."""
    # untreated: shunt worsens, compliance drops
    h.shunt_fraction    += 0.002 * severity * (dt_s / 60)
    h.compliance_index  -= 0.005 * severity * (dt_s / 60)
    # tachycardia from hypoxic stress
    h.chronotropic_drive += 0.01 * severity * (dt_s / 60)

    # Tipping point: compensation fails → ventilatory collapse
    # (this captures "patient tires" transitions that pure linear drift misses)
    if h.chronotropic_drive > 2.0 and h.compliance_index < 0.5:
        h.ventilatory_drive = max(0.3, h.ventilatory_drive - 0.05 * (dt_s / 60))
        h.consciousness     = max(0.0, h.consciousness - 0.02 * (dt_s / 60))
```

One rule per pathology. Rule body is pure: takes `(hidden, severity, dt)`,
mutates `hidden` in place. No access to case_id, actions, or other context —
this is what enforces generalization.

**Pathology rules may be piecewise, not just linear drift.** Scenarios frequently
show "tipping point" transitions (patient tires, decompensates, arrests) that
can't be captured by monotonic linear drift. Rules are allowed — encouraged —
to branch on hidden-state thresholds to produce these non-linear transitions.
The constraint stays the same: branching must be on **hidden state**, never on
`case_id` or scenario metadata.

**Drift rates must be scenario-calibrated, not physiological.** Scenarios are
time-compressed teaching tools: severe pulmonary edema visibly worsens over
5 minutes in-script, whereas real-world drift over the same window would be
smaller. Phase 4 calibration uses the 249 pure-wait pairs to fit
**scenario-paced** drift rates, not ICU-derived ones. The goal is reproducing
author intent, not real physiology.

**Phase 4 accepted patterns (post-validation 2026-04-19):**

- **Pure-wait gate in the integrator** (`engine.py`): drift rules fire only
  when `len(actions) == 0`. This preserves Phase 3b byte-identical alignment
  on action pairs. Phase 5 removes this gate once intervention × drift
  composition is implemented.
- **Severity ≥ 0.9 gate inside each rule**: drift applies only when
  `severity_to_scalar(pathology.severity) >= 0.9` (i.e. "severe"). Author-
  stable majority of mild/moderate pairs is preserved. Rules may make this
  stricter but not looser.
- **Per-rule net-impact validation before registration**: before merging a
  drift rule, run it over its pathology's pure-wait subset and verify net
  improvement in per-vital MAE. Phase 4 found 9 candidate rules net-negative
  (adrenal_crisis, airway_obstruction, aortic_dissection, asthma_exacerbation,
  neonatal_respiratory_distress, pea_arrest, septic_shock, stemi, vf_arrest)
  — these are pinned as no-op until Phase 5 intervention composition gives
  them a counter-force. Do not "just ship" a rule that fails this check.
- **No DEFAULT_DRIFT fallback.** Unregistered pathologies drift not at all.
  A generic fallback was validated net-negative across the long tail.
  Phase 7 outlier triage handles individually.
- **Tipping points branch on hidden-state thresholds** (§5 is explicit on
  this): e.g. `if h.preload_index < 0.3 and h.CO_index < 0.5: h.rhythm = "PEA"`.
  Never branch on case_id / scenario metadata.

### Intervention (discrete jumps at t=0)

```python
# intervention_lib.py
INTERVENTION_RULES = {}

@intervention_rule("apply_NRB")
def apply_nrb(h, flags, params):
    flags["O2_device"] = "NRB"
    flags["FiO2"] = 1.0
    # PaO2 rises toward alveolar O2, attenuated by shunt
    alveolar_PaO2 = 100 + 600 * (flags["FiO2"] - 0.21)
    h.PaO2_effective = lerp(h.PaO2_effective, alveolar_PaO2,
                             weight = 1.0 - h.shunt_fraction)

@intervention_rule("intubate")
def intubate(h, flags, params):
    flags["intubated"]  = True
    flags["O2_device"]  = "vent"
    flags["vent_rate"]  = flags.get("vent_rate") or 12
    flags["vent_TV_ml"] = flags.get("vent_TV_ml") or 500
    # NOTE: do NOT set flags["airway"] = True. ETT is captured by `intubated`,
    # not `airway`. `airway` is reserved for OPA/NPA/non-ETT adjuncts. Verified
    # against dataset: 74/78 intubate pairs leave airway=False. See
    # transitions/EXTRACTION_GUIDE.md §8.
    # mechanical ventilation recruits alveoli → shunt improves
    h.shunt_fraction *= 0.6
    h.compliance_index = max(h.compliance_index, 0.7)
    # NOTE: we deliberately do NOT model sedation-driven HR drop here
    # (per design decision — requires explicit drug action in scenario)
```

**Surgical airway interventions are NOT equivalent.** Data-driven distinction:

- `needle_cricothyroidotomy` (temporizing cannula) → sets `airway=True` only.
  No mechanical ventilation circuit — patient breathes through cannula.
- `surgical_cricothyrotomy` (definitive surgical ETT) → same flag set as
  `intubate` (`intubated=True`, `O2_device='vent'`, `FiO2=1.0`, `vent_rate` /
  `vent_TV_ml` defaults if `None`). `airway` stays `False` — ETT is not an
  adjunct.
- `escharotomy` (burn contracture release) → no airway-flag effect at all;
  purely hidden-state (chest compliance in Phase 5).

Rule registrations must keep these three distinct — the brief's earlier
grouping as "三个外科气道 → `intubated=True` + `airway=True`" was incorrect.

### Drug (time-dependent PD, class-based)

**Phase 6 (current)**: drugs are applied at `t=0` via `drug_lib.start_drug`,
which enqueues one `DrugEffect` per class-template entry onto
`h.active_drug_effects`. `drug_lib.tick` advances each effect's PK clock every
integration tick and applies the delta contribution to its target hidden var.
The `DRUG_*` registries below are populated (14 classes, 54-drug map, ~14
per-drug overrides).

Drug memory is **per-step** only: at L1 the engine decodes a fresh
`HiddenState` from `before` on every `step()` call, so drugs expire at the
pair boundary. Cross-pair persistence is deferred to L2 Session (§6.5). The
corpus has no `state.drugs[]` field — drugs appear only in `pair.actions[]`.

**Strategy: drugs are grouped into pharmacologic classes.** All drugs in a class
share the same PD template; individual drugs only differ by dose magnitude and
route-specific kinetics. This collapses 54 drugs × ~2 effects (~100 rows to
hand-author) down to ~12 classes × ~2 effects (~25 rows) plus a simple `drug →
class` lookup.

```python
# drug_lib.py

# --- Class templates: {class_name: PD spec} ---
DRUG_CLASSES = {
    "pressor_alpha_beta": {           # epi, high-dose dopamine
        "effects": [
            {"target": "chronotropic_drive", "magnitude": +1.5, "half_life_s": 180},
            {"target": "SVR_index",          "magnitude": +0.4, "half_life_s": 180},
            {"target": "CO_index",           "magnitude": +0.3, "half_life_s": 180},
        ],
        "onset_s": 30,
    },
    "pressor_alpha_pure": {           # norepi, phenylephrine, vasopressin
        "effects": [
            {"target": "SVR_index", "magnitude": +0.5, "half_life_s": 120},
        ],
        "onset_s": 60,
    },
    "inotrope_beta":     { ... },     # dobutamine, milrinone
    "sedative":          { ... },     # propofol, midazolam, etomidate, ketamine
    "paralytic":         { ... },     # rocuronium, succinylcholine, vecuronium
    "antiarrhythmic_3":  { ... },     # amiodarone, sotalol
    "vagolytic":         { ... },     # atropine, glycopyrrolate
    "bronchodilator":    { ... },     # albuterol, ipratropium
    "electrolyte":       { ... },     # Ca gluc, Mg, NaHCO3, K
    "anticonvulsant":    { ... },     # lorazepam, phenytoin, levetiracetam
    "opioid":            { ... },     # fentanyl, morphine
    "reversal":          { ... },     # naloxone, flumazenil
    # ~12 classes cover ~95% of 54 drugs
}

# --- Drug → class map ---
DRUG_TO_CLASS = {
    "epinephrine":        "pressor_alpha_beta",
    "norepinephrine":     "pressor_alpha_pure",
    "phenylephrine":      "pressor_alpha_pure",
    "vasopressin":        "pressor_alpha_pure",
    "dobutamine":         "inotrope_beta",
    "propofol":           "sedative",
    "midazolam":          "sedative",
    "atropine":           "vagolytic",
    "amiodarone":         "antiarrhythmic_3",
    "rocuronium":         "paralytic",
    # ... 54 entries total
}

# --- Optional per-drug overrides for the few that misbehave ---
DRUG_OVERRIDES = {
    "ketamine": {                     # sympathomimetic sedative — unusual
        "extra_effects": [
            {"target": "chronotropic_drive", "magnitude": +0.3, "half_life_s": 300},
            {"target": "SVR_index",          "magnitude": +0.1, "half_life_s": 300},
        ],
    },
    # ... a handful of special cases
}

# --- Dose scaling ---
# Per-class standard dose defined; actual dose / standard dose = magnitude multiplier
DRUG_STANDARD_DOSE = {
    "epinephrine":    {"IV": 1.0, "unit": "mg"},   # 1 mg IV push in arrest
    "norepinephrine": {"IV_infusion": 0.1, "unit": "mcg/kg/min"},
    # ...
}
```

Effect magnitudes scale linearly with dose ratio (`actual / standard`), clipped
to `[0.1x, 3x]` to avoid absurd extremes.

**Confidence tagging and medical review.** Every class template is tagged in a
comment with `confidence: high/med/low` and `source: <citation or "author">`.
Before Phase 6 runs, the class templates get the same 30-min clinician review as
the mapping module — **direction signs** must be correct before magnitudes are
fit against data. Author-drafted magnitudes are starting points, not ground truth.

Hand-authoring ~25 rows is ~1-2 hours; reviewing is ~30 min. This is the cheapest
viable path to covering 54 drugs.

## 6. Integrator

```python
def step(before_dict, actions, duration_s):
    h, flags = io.decode_state(before_dict)

    # --- t=0: apply all actions (discrete jumps) ---
    for action in actions:
        if action["type"] == "intervention":
            INTERVENTION_RULES[action["name"]](h, flags, action)
        elif action["type"] == "drug":
            drug_lib.start_drug(h, action)   # schedules DrugEffect

    # --- t>0: integrate pathology + drug PD ---
    dt = 30.0  # seconds per tick
    t = 0
    T_before = before_dict["vitals"]["T"]
    while t < duration_s:
        step_dt = min(dt, duration_s - t)
        pathology = before_dict["mechanism"]["pathology"]
        PATHOLOGY_RULES[pathology["name"]](h, severity_to_scalar(pathology["severity"]), step_dt)
        drug_lib.tick(h, step_dt)
        t += step_dt

    # --- encode back to schema state ---
    return io.encode_state(h, flags,
                           T_before=T_before,
                           dt_s=duration_s,
                           before_dict=before_dict)
```

Internal Δt = **30s** (design freeze). When `duration_s` is not a multiple of
30, the last tick uses the remainder (e.g. 45s → 30s + 15s). 30s balances PK
capture (epi peak ~60s) against compute cost and was **explicitly decided over
5s**. Rules receive the actual `step_dt` each call and must use it to scale
their effects — a rule that writes `h.x += 0.1` would have behaved differently
under the old 5s cadence; rules that write `h.x += 0.1 * (dt_s / 60)` are
cadence-independent.

Key points:
- Intervention rules fire **once at t=0** (discrete jumps)
- Pathology + drug rules fire **every tick** (continuous drift)
- `dt = 30s` is the integration step; pair-level `duration_s` ranges 0-600s
- When `duration_s = 0`, the loop is skipped — only intervention jumps apply

### 6.1 `encode_state` contract

```python
def encode_state(
    h: HiddenState,
    flags: dict,
    *,
    T_before: float,
    dt_s: float,
    before_dict: dict,         # full before state, for pass-through fields
) -> dict:
    ...
```

Output fields: `vitals` computed from `vitals_from_hidden(h)`; `T_before` and
`dt_s` are load-bearing for `T_from_hidden(h, T_before, dt_s)`; `interventions`
from the mutated `flags` dict; `mechanism.pathology` passes through from
`before_dict` (input, not engine-owned). There is no `drugs` field in the
output — drugs live only in `pair.actions[]` in the corpus, and within-pair PD
lives on `h.active_drug_effects`.

## 6.5. Agent & Orchestrator Interface

The engine has three API surfaces:

```
┌───────────────────────────────────────────────────┐
│  L3: Orchestrator protocol  (interface only in v1)│
├───────────────────────────────────────────────────┤
│  L2: Agent-facing API       (stubbed in v1)       │
├───────────────────────────────────────────────────┤
│  L1: Core transition API    (fully in v1)         │
└───────────────────────────────────────────────────┘
```

L1 is what gets evaluated on the 808 pairs. L2 and L3 are where LLM agents will
plug in later. We define all three shapes now so the engine's internal state
machine is agent-ready from day one, even if we only implement L1.

### L1 — Core transition API (v1 full)

```python
engine.step(before: dict, actions: list[dict], duration_s: float) -> dict
```

Stateless, deterministic, teacher-forced. This is what `emsim_eval.py` uses.
Self-contained enough to be tested in isolation on the 808 pairs.

### L2 — Agent-facing API (v1 stubbed)

Agent loops need a stateful, observation-oriented API. The engine holds a
persistent `SessionState` and exposes verbs the orchestrator can call:

**Why L2 is stateful, not just a thin L1 wrapper.** L1's re-decode-per-call
is the right semantics for independent-pair eval but the wrong semantics
for trajectories: `decode_state` is lossy on hidden vars without observable
vitals (see §10). L2 owns `self._h` as the canonical hidden state across
an entire simulation, so drug PD decays, pathology drift accumulators, and
unobservable physiology (consciousness, temperature trend) remain
continuous across agent interactions. Concretely: `submit_action()` mutates
`self._h` in place; `advance()` integrates `self._h` forward; `observe()`
reads a snapshot. Never call `decode_state` during a live session.

```python
class EngineSession:
    """Stateful wrapper around the core engine for agent loops."""

    def __init__(self, initial_state: dict):
        self._h, self._flags = io.decode_state(initial_state)
        self._t = 0.0  # sim seconds elapsed
        self._T_last = initial_state["vitals"]["T"]
        # Drug memory lives on self._h.active_drug_effects — L2 persists it
        # across submit_action / advance calls (L1 resets it per step()).
        self._event_queue = []    # events since last observe()

    # --- Actions ---
    def submit_action(self, agent_id: str, action: dict) -> ActionReceipt:
        """Apply action immediately (t=0 jump). Returns what happened."""
        ...

    # --- Time advance ---
    def advance(self, duration_s: float) -> list[Event]:
        """Integrate drift + drug PD for duration_s. Returns emitted events
        (arrest, ROSC, seizure, etc). MAY return early if a critical event fires."""
        ...

    # --- Observations ---
    def observe(self, agent_id: str) -> Observation:
        """Return what this agent can perceive right now. Different agents
        see different views of the same underlying state (see L3)."""
        ...

    # --- Introspection (for debugging / logging, not for agents) ---
    def snapshot(self) -> dict:
        """Full schema state dict, same format as before/after in pairs."""
        ...
```

`ActionReceipt` = `{success: bool, message: str, immediate_vital_change: dict}`.
E.g. defibrillating a non-shockable rhythm returns `success=False, message="rhythm
is asystole, not shockable"`. This is how agents get action-legality feedback.

`Event` is emitted by the engine when something state-changing happens that agents
didn't cause:
```python
@dataclass
class Event:
    kind: str       # "arrest", "rosc", "seizure", "arrhythmia_change",
                    # "vital_alarm", "compensation_failure"
    t_sim_s: float  # when it happened in simulation time
    details: dict   # rhythm change, which vital alarmed, etc.
```

Events are how the "tipping point" logic from §5 (pathology tires → ventilatory
collapse) surfaces to the outside world. The rule layer emits events; the
session collects them; `advance()` returns them so the orchestrator can interrupt.

### L3 — Orchestrator protocol (v1 interface only)

The orchestrator is a separate component (not part of the engine). It sits
between multi-agent turns and the engine session, and decides things the engine
shouldn't decide:

```python
class OrchestratorProtocol(Protocol):
    def next_turn(self, session: EngineSession) -> AgentTurn:
        """Decide which agent acts next, or whether to advance time."""

    def dispatch_events(self, events: list[Event]) -> None:
        """Route engine events to relevant agents (e.g. arrest → clinician)."""

    def filter_observation(self, agent_id: str, full_state: dict) -> dict:
        """Project full state to what this agent can perceive. See table below."""
```

The **filter_observation** function is what makes different agent roles see
different things:

| Agent         | Sees                                                      | Does NOT see                            |
|---------------|-----------------------------------------------------------|------------------------------------------|
| Clinician     | Monitor vitals, interventions in place, drugs given      | Hidden state; patient's subjective experience |
| Nurse         | Same as clinician, plus real-time alarms                 | Same                                    |
| Patient       | Subjective symptoms (dyspnea, pain, consciousness)       | Their own vitals; their own pathology name |
| Relative      | Patient's apparent distress; clinician-delivered updates | Anything not said out loud              |

In v1 we don't implement any of this filtering — but the shape of
`filter_observation` is defined so the engine knows to store information the
filter will need (e.g. `consciousness` so the patient agent can be scripted with
"you are confused"; `subjective_dyspnea` derived from PaO2 + consciousness).

### What v1 actually builds

Concretely in `rule_engine/`:

- `engine.py` — implements L1 fully (the `RuleEngine` class with `.step()`)
- `session.py` — implements L2 as a **thin wrapper** around L1:
  - `submit_action()` → delegates to `step()` with `duration_s=0`
  - `advance(t)` → delegates to `step()` with `actions=[]`, `duration_s=t`
  - `observe()` → returns full `snapshot()` (no filtering yet)
  - `_event_queue` stays empty in v1 (no event emission yet)
- `orchestrator.py` — **defines Protocol, no implementation.** Lets future
  code import the types.
- Event classes defined in `events.py`, unused but importable.

This means: **agents can't actually connect yet**, but the moment someone writes
an orchestrator, every verb it needs already exists on `EngineSession`. No
refactor required.

### Engine scope boundary (what engine does and does not own)

The cleanest way to think about this:

> **Engine only cares about: what the action is, what state it acts on,
> and what the resulting state is.**
>
> Everything else — who picked the action, why, whether it should have been
> allowed, how it interacts with other agents' actions — is someone else's job.

Concretely:

| Belongs to engine                          | Belongs OUTSIDE engine (agent layer / orchestrator) |
|--------------------------------------------|-----------------------------------------------------|
| Hidden-state physics                       | Agent action spaces (what a nurse is allowed to do) |
| Pathology drift rules                      | Agent decision logic (LLM prompting, tool use)      |
| Intervention and drug effects              | Agent memory systems                                |
| Emitting physiologic events                | Action conflict resolution between agents           |
| Returning full `snapshot()`                | Per-agent observation filtering                     |
| `ActionReceipt` (was the action valid physiologically?) | Whether the agent had the authority/skill to do it |

Engine treats an incoming action as a **well-formed instruction from the outside
world**. It doesn't ask who sent it. `agent_id` is an opaque tag the engine logs
but never routes on. This keeps the engine's API surface stable even as the
agent layer evolves freely.

### What's locked vs what's flexible

Three things are architecturally fixed in v1 — changing them later costs a
rewrite:

1. **L1 is stateless.** `step(before, actions, duration_s) → after`. This is
   what the 808-pair evaluation depends on.
2. **L2 is the stateful layer.** Persistent hidden state, drug timelines, and
   accumulated events live only in `EngineSession`, never in L1.
3. **Engine doesn't know about agents.** `agent_id` is a string tag, not a
   routing key. No per-agent state lives inside the engine.

Everything else in L2 / L3 is deliberately under-specified and easy to change
later. The following are **starting defaults**, not commitments — any of them
can be swapped out without touching L1 or the rule libraries:

- *Default:* `advance()` may return early on critical events → *swap to:* always
  run to completion and batch-return events. One method body.
- *Default:* conflicting actions applied in submission order (last wins) →
  *swap to:* reject all conflicts / priority-based / time-slicing. Orchestrator
  concern anyway.
- *Default:* `observe()` returns full `snapshot()` →
  *swap to:* per-agent filtered views. Orchestrator concern; engine unchanged.
- *Default:* `Event` has `{kind, t_sim_s, details}` →
  *swap to:* richer schema with severity, source rule, suggested responses.
  Just the dataclass.
- *Default:* in-memory session only → *swap to:* pickleable / checkpointable.
  Extend `EngineSession`; L1 untouched.

### Other v1 design decisions

- **Engine is deterministic.** No stochasticity in v1 even in the agent API.
  Distributions are Phase 2 of the whole project.
- **Session state is in-memory, not persisted.** Checkpointing / replay is
  a separate concern, out of v1 scope.

## 7. Phased Build Plan

Each phase ends with a concrete metric target. If the target isn't hit, diagnose
before moving on.

### 7.0 A word on Phase targets (pattern observed in Phases 2, 4, 5)

**Targets are lower bounds on contribution, not independent upper bounds.**
Phases 2, 4, and 5 all initially set targets that assumed a single Phase's
new capability could independently push metrics to a threshold. In each case
the data revealed that the metric shift was actually **composition-gated**
on a later Phase:

- **Phase 2** (rhythm + arrest mapping) initial target HR MAE ≤ 11 was
  unreachable because most arrest failures need Phase 4 tipping points to
  predict arrest emergence.
- **Phase 4** (pathology drift) initial target pure-wait strict 30%+ was
  unreachable because ~230/249 pure-wait pairs are author-stable and any
  linear drift loses dir_match.
- **Phase 5** (intervention effects + drift composition) initial target
  action strict 40%+ was unreachable because 261/372 failures are multi-
  dimensional and need Phase 6 drug PD to close HR+BP+RR simultaneously.

**Working principle going forward:** each Phase target is the **infrastructure
delivery criterion** (worst-20 focus transfers, regression bounded, per-slice
improvement on its own subset), not the headline strict number. The strict
headline is a composition of all Phases and only materializes at the end of
the chain. This is why Phase 6 has the largest projected jump — it's the
Phase that finally lets all prior infrastructure compose.

### Phase-level metric log (for sanity)

| Phase | Strict | Partial | Notes |
|---|---|---|---|
| 0 (skeleton) | 13.9% | 0.855 | drug plumbing only |
| 1 (flag + drug plumbing) | 22.6% | 0.880 | intervention flags registered |
| 2 (rhythm infrastructure) | 22.6% | 0.880 | no metric change (as designed) |
| 3a (decode + mapping calibration) | 22.6% | 0.880 | round-trip 97% |
| 3b (encode → from_hidden) | 22.0% | 0.882 | RR MAE -0.74 (intubate sync) |
| 4 (pathology drift infrastructure) | 22.4% | 0.882 | drift rules + 8 tipping points |
| 5 (intervention + drift composition) | 23.3% | — | per-intervention win, multi-dim failures bounded |
| 6 (drug PD — projected) | 38-45% | — | composition-driven jump |
| 7 (outlier triage — projected) | +2-3pp | — | freeze canonical number |


### Phase 0 — Skeleton (0.5 day)
Build: file structure, empty dataclass, empty registries, engine `.step()` that
decodes/encodes identity state.
**Target:** parity with `IdentityEngine` (2.7% strict, 0.848 partial).

### Phase 0.5 — Agent API stubs (0.5 day)
Build: `session.py` (thin wrapper over `.step()`), `events.py` (dataclasses),
`orchestrator.py` (Protocol only). Add integration test: open a session, submit
an action, advance 60s, observe — verify `snapshot()` matches what `.step()`
would have produced for the equivalent pair.
**Target:** L1-L2 round-trip test passes. L1 eval metrics unchanged.

### Phase 1 — Flag plumbing (0.5 day)
Build: intervention rules that only touch `flags` dict (no hidden state effects yet).
Drug actions are consumed by `drug_lib.start_drug` at t=0 (no `drugs[]` field
in state after the 2026-04-19 removal; see DRUG_REMOVAL_REPORT.md).
**Target:** `interv_match` 56% → 85%+. Vital MAE unchanged.

### Phase 2 — Rhythm infrastructure + defib transition (1 day)
Build: `rhythm` field inference in `decode_state` from `before` state (HR /
CPR_active / pathology name); arrest-rhythm branch in `HR_from_hidden` and
`BP_from_hidden` returning 0 (monitor semantics); `O2Sat_from_hidden` returns
None for pass-through (see §4 note — pulse-ox holds last valid reading on
arrest, not a true 0). `defibrillate` flips `h.rhythm` from VF → sinus at t=0
(no-op on asystole/PEA per clinical consensus).

**Target: zero metric gain expected.** Phase 2 is prerequisite scaffolding,
not an eval-moving phase. Rationale (verified on 2026-04-18 Phase 2 run):
- Of 57 after-arrest pairs, 38 already have before.HR==0, so Phase 1's
  pass-through already outputs 0 — no room to improve via arrest-zeroing.
- The remaining 19 are arrest-emergence pairs (before.HR>0 → after.HR=0).
  These cannot be predicted from before-state alone without false positives;
  they require **Phase 4 pathology drift with tipping points**.
- Defibrillate's VF→sinus transition can't surface in vitals until Phase 3
  calibrates the sinus HR mapping (currently pass-through returns 0).

**Real MAE improvement targets are deferred to Phases 3-4**:
- Phase 3 (mapping calibration): enables defib transitions and non-arrest
  sinus HR to produce sensible values.
- Phase 4 (pathology drift): enables arrest-emergence prediction for pairs
  where `before` shows deterioration precursors.

**Phase 2 acceptance criteria** (not metric-based):
- All 57 after-arrest pairs with before.HR==0 classified into a valid arrest
  rhythm (VF / PEA / asystole) by `infer_rhythm`.
- Non-arrest per-vital MAE byte-identical to Phase 1 (mapping doesn't
  leak into non-arrest pairs).
- Rhythm distribution sanity check across 808 before-states (expect
  ~90% sinus, rest split among arrest/VT/SVT/brady).

### Phase 3 — Vitals mapping calibration (1-2 days)
**Critical checkpoint.** For every pair: decode `before` → map → check that
predicted `before.vitals` matches actual `before.vitals` within tolerance.
Tune baseline coefficients (75 in HR, 90 in MAP, hemoglobin curve anchors) until
this holds for ≥ 95% of pairs.
**Target:** `decoder→mapper` round-trip MAE < tolerance/2 on all 6 vitals.

### Phase 4 — Pathology drift infrastructure (2-3 days)

Build drift rules for the ~30 unique pathology names in the dataset, plus
registration machinery, tipping-point primitives, and drift-vs-vitals
diagnostic tooling. Use pure-wait drift statistics as priors:
- HR: more often up-then-down (failure pattern)
- BP: 2:1 bias toward down
- O2Sat: 5:1 bias toward down untreated
- RR: mostly stable
- T: mostly stable

**Realistic target (revised after Phase 4 validation, 2026-04-19):**
pure-wait strict 7.6% → 10-12%. Pure-wait HR MAE → ≤ 18. O2Sat MAE → ≤ 10.
Action-pair metrics byte-identical to Phase 3b.

**Why the original 30% target was unreachable:** 249 pure-wait pairs decompose
as:
- ~13 arrest-emergence (require scenario-specific tipping — hidden-state
  thresholds alone underfit)
- ~3 arrest-resolution (no in-pair signal to predict ROSC)
- ~230 author-stable pairs where ground truth vitals move minimally (often
  inside the direction dead-zone, HR±3 / O2Sat±1). Any linear drift pushes
  predictions out of the dead-zone and loses `dir_match`, even when
  magnitude MAE is fine.

The 230-pair stable majority is a **scenario-authoring reality**: authors
write "observation window, no change" pairs to pace the learner, not to
model continuous decline. Linear drift on all pathologies would regress
more pairs than it fixes; per-rule impact tests in Phase 4 validated 9
rules as net-negative and pinned them to no-op.

**Accepted Phase 4 implementation patterns:**
- **Pure-wait gate**: drift rules execute only when `actions=[]`. Action
  pairs stay Phase 3b byte-identical. Phase 5 will unlock this once
  intervention×drift composition works.
- **Severity ≥ 0.9 gate**: drift fires only on `severe` pathology; mild /
  moderate are treated as stable. Author-stable majority is preserved.
- **Tipping points branch on hidden-state thresholds only** (no scenario
  metadata): e.g. `h.preload_index < 0.3 and h.CO_index < 0.5 → rhythm=PEA`.
  Captures arrest emergence when the drift has pushed hidden state past a
  decompensation threshold.
- **DEFAULT_DRIFT neutralized** (no generic fallback): unregistered
  pathologies drift not at all. Phase 7 outlier triage handles long tail.

**Phase 4 acceptance criteria (not primarily metric):**
- Pure-wait strict ≥ 10% (realistic ceiling given author-stable majority).
- Action non-defib strict byte-identical to Phase 3b.
- Round-trip ≥ 95%.
- Drift rule registry ≥ 40 active rules covering all pathologies with ≥ 5
  pure-wait pairs.
- Tipping-point logic present for the top arrest-emergence pathologies
  (aortic_dissection, stemi_with_bradycardia pattern, tba).
- T-drift safety net removable (core_temp_trend driven by drift rules only).

### Phase 5 — Intervention vital effects + drift composition (2-3 days)

This is where the real metric lift happens. Two entangled objectives:

**(a) Intervention hidden-state effects.** Fill out the hidden-state
mutations in each of the 21 intervention rules (beyond the flag mutations
from Phase 1). Priorities by frequency in dataset:
1. O2 escalation: `apply_nasal`, `apply_NRB`, `apply_BVM`, `add_PEEP`,
   `intubate` — each raises `h.PaO2_effective` toward `alveolar_PaO2`,
   attenuated by `h.shunt_fraction`
2. Perfusion: `give_fluids` (preload), `start_CPR` (forced CO during
   compressions), `defibrillate` (rhythm conversion)
3. Chest: `needle_decompress`, `place_chest_tube` — restore venous return
   and ventilation compliance

**(b) Unlock drift on action pairs.** Phase 4 gated drift to `actions=[]`
to preserve Phase 3b's byte-identical action alignment. Phase 5 removes
that gate once intervention hidden-state effects can *counter* the drift.
E.g. ARDS drift progressively raises shunt_fraction; `apply_NRB` raises
PaO2_effective — in composition, the net O2Sat change depends on which
side wins. Composition is the physics Phase 5 captures.

**Sequencing recommendation:** do (a) first to get intervention effects
registered, then carefully unlock the drift gate in (b) while watching
the action non-defib MAE as canary. If composition regresses specific
pathology+intervention pairs, tune the intervention magnitudes before
tuning drift — drift was calibrated on pure-wait and shouldn't move.

**Realistic target (revised after Phase 5 validation, 2026-04-19):**
action-pair strict 29.3% → 30-32%. Action O2Sat MAE 5.32 → ≤ 4.
Overall strict 22.4% → 23-24%.

**Why the original 40%+ target was unreachable:** Phase 5 validation showed
that 261 of 372 remaining action_non_defib failures are multi-dimensional
(HR + BP + RR all off by > tolerance simultaneously). These failures require
drug PD composition — most "intubation" pairs carry concurrent propofol +
rocuronium, "shock" pairs carry norepinephrine/epinephrine, "agitation"
pairs carry midazolam. Without Phase 6 drug-class templates, these pairs
are bounded by "I can move one vital with the intervention but not the
other two" — a structural metric ceiling, not a calibration gap.

The Phase 5 architectural work was the real deliverable:
- Every intervention now has hidden-state effects (not just flag mutations).
- Drift gate unlocked on intervention pairs; drug-only pairs still gated
  (Phase 6 flips this once drug PD lands).
- 5 of 9 pinned drift rules un-pinned (adrenal_crisis, asthma_exacerbation,
  septic_shock, stemi, aortic_dissection) after validation showed net-positive
  composition behavior.
- Worst-20 focus transferred entirely — 0 worst-20 pairs are now
  "intervention effect wrong"; remaining failures are arrest-emergence,
  arrest-resolution, data-quality, and drug-PD-missing.

**Phase 5 acceptance criteria (not primarily strict target):**
- Every intervention shows per-intervention strict improvement on its
  single-action pair subset (e.g. apply_nasal 12.5% → 59.4%, apply_BVM
  4.5% → 36.4%, apply_NRB 25% → 41.7%).
- Pure-wait strict does not regress (≥ 7%).
- Round-trip ≥ 95%.
- Worst-20 analysis shows 0 pairs flagged as "intervention effect wrong".
- Diagnostic tooling in place: `_intervention_test.py` (per-intervention
  slices + worst-20 per slice), `_pinned_rules_test.py` (drift body A/B).
- EMSIM_DRIFT_GATE=1 env var flips gate back to Phase 4 for A/B testing.

### Phase 6 — Drug PD + multi-action composition (2-3 days)

**This is the final big metric jump.** Fill the ~12 drug-class templates
in `DRUG_CLASSES`, populate the 54-entry `DRUG_TO_CLASS` map, author
per-drug overrides (ketamine etc.), and flip the drug-only drift gate
so drug pairs benefit from pathology drift composition.

**Start with the classes covering the top-10 most-used drugs** (epinephrine,
midazolam, norepinephrine, calcium_gluconate, propofol, atropine, amiodarone,
NaHCO3, magnesium, rocuronium) — these cover 60% of drug uses and span
~6 classes (pressor_alpha_beta, sedative, antiarrhythmic_3, electrolyte,
vagolytic, paralytic).

**Why this is where the big jump happens** (composition logic from Phase 5
analysis): 261/372 action_non_defib failures are multi-dimensional — an
intubation pair sets O2 correctly via Phase 5 but misses the sedative-
driven HR drop and paralytic RR floor. Adding drug PD closes all three
dimensions simultaneously on the dominant failure cluster.

**Realistic target:** overall strict 23% → 38-45%. Action non-defib strict
31% → 50%+. HR MAE → ≤ 8. BP_sys MAE → ≤ 10. Drug-only pair strict (not
yet tracked) should exit baseline and become meaningful.

**Phase 6 acceptance criteria:**
- 54 drugs → 12 class templates fully mapped (no unmapped drugs with >1
  use in corpus).
- Top-10 drugs show per-drug slice improvement.
- Drug-only drift gate unlocked; pure-wait / intervention strict not
  regressed.
- Worst-20 focus transfers to "arrest emergence" and "data-quality"
  categories (Phase 7 triage territory).
- Diagnostic `_drug_test.py` in place (analog to `_intervention_test.py`).

### Phase 7 — Outlier triage + final baseline (0.5 day)
Review the 11 high-priority issues flagged in PROJECT_STATUS.md. Decide per-case:
fix the scenario JSON, or mark the pair as a known-outlier excluded from the
canonical pass rate. Rerun and freeze v1 numbers.

**Target document:** `v1_baseline.md` with final per-category pass rates + known
outliers list.

## 8. Calibration Strategy

Three parameter classes, three strategies:

1. **Structural constants** (hemoglobin curve anchors, baseline HR/MAP, Frank-Starling
   curve shape): fit in Phase 3 by minimizing vital reproduction error across
   all 808 `before` states. These are shared across all cases.

2. **Pathology drift rates**: fit per-pathology using only pure-wait pairs where
   that pathology is active. Minimize vital MAE on `after` given `before` and `duration_s`.

3. **Intervention & drug magnitudes**: fit against single-action pairs where that
   specific intervention/drug is the only action. Reserve multi-action pairs for
   composition validation — we should never fit magnitudes on them, only verify.

Leave-out set: pick ~10% of cases (by case_id, not by pair) to exclude from all
fitting. Report final metrics separately on the held-out set.

## 9. Iteration Loop

After every rule addition:
```bash
python emsim_eval.py --transitions transitions \
    --engine rule_engine:RuleEngine \
    --export results.csv --worst 20
```
Then:
1. Check headline: partial score trending up, per-vital MAE trending down
2. Diff `results.csv` vs previous run — any regressions on pairs that were passing?
3. If yes, the new rule has unintended interaction → narrow its condition
4. Check worst-20 list — is it now dominated by a new failure mode? That's the
   next rule to write.

## 10. Explicit Non-Goals for v1

These are deferred or permanently out-of-scope. Don't sneak them in:

- **Implicit co-interventions.** Intubation does NOT auto-trigger sedation
  effects on HR. Scenarios that show peri-intubation HR drops need explicit
  drug actions in the JSON, or will be accepted as outliers.
- **Drug-drug interactions.** Each drug's PD is independent; effects on the same
  hidden var simply add.
- **Age-dependent physiology.** Peds cases will use the same baselines.
- **Lab values** (glucose, K+, pH, troponin). Not in the 6 vitals, not modeled.
- **Stochasticity.** v1 is deterministic; distribution outputs are Phase 2 of
  the whole project.
- **Rollout evaluation** (chained pairs). Still teacher-forced per-pair in v1.
  When rollout is added, it MUST go through L2 `EngineSession` (§6.5) — do
  NOT chain L1 `step()` calls. L1 is stateless by design: each call
  re-derives HiddenState from the observable schema via `decode_state`,
  which is lossy for hidden vars that have no direct vital counterpart
  (consciousness, CO/SVR joint allocation, shunt/compliance joint
  allocation, `core_temp_trend`, drug PD magnitude dynamics). Chaining L1
  silently collapses every step back toward the pathology signature
  baseline. L2 persists `self._h` across advances and is the only correct
  rollout path.

## 11. LLM Baseline Comparison

In parallel with building the rule-based engine, we evaluate a **pure LLM engine**
on the same 808 pairs, using the same harness. The goal is a controlled
rule-based vs LLM comparison that tells us:

1. Where rule-based wins (stable, cheap, reproducible on structured drift)
2. Where LLM wins (medical reasoning, atypical cases, narrative-coherent trajectories)
3. Whether a future hybrid — LLM for judgment + rules for mechanics — could
   outperform either alone

### Interface

LLM engine implements the same `PhysiologyEngine` protocol:

```python
class LLMEngine:
    def __init__(self, model: str, prompt_variant: str):
        self.model = model
        self.prompt_variant = prompt_variant  # "zero_shot" | "few_shot" | "with_case_context"

    def step(self, before: dict, actions: list[dict], duration_s: float) -> dict:
        prompt = self._build_prompt(before, actions, duration_s)
        delta = self._call_model(prompt)  # LLM returns delta JSON
        return self._apply_delta(before, delta)
```

Plugs into existing eval harness: `python emsim_eval.py --engine llm_engine:LLMEngine`.

### Output format — deltas, not full states

The LLM returns a **delta** over `before`, not a regenerated `after`:

```json
{
    "vitals_delta":          {"HR": +10, "O2Sat": -3, "RR": -10},
    "interventions_changed": {"O2_device": "NRB", "FiO2": 1.0},
    "mechanism_severity":    "severe",
    "reasoning":             "Patient applied NRB → O2Sat improved from 80 to 88..."
}
```

Why delta, not full state:
- **Most fields don't change** — 808 pairs have ~80% of state keys unchanged on
  average. Regenerating everything wastes tokens and introduces drift on fields
  that should stay put.
- **Easier to validate** — easier to check "did LLM only claim to change X, Y, Z"
  than to diff full state objects.
- **Reasoning field is optional** but highly useful for error analysis.

The `_apply_delta()` helper takes `before` + delta → full valid `after` dict.

### Prompt variants (each one a separate run)

| Variant                | Context given                                          | Answers               |
|------------------------|--------------------------------------------------------|-----------------------|
| `zero_shot`            | Schema + `before` + `actions` + `duration_s`           | LLM medical reasoning floor |
| `few_shot`             | Above + 3-5 example pairs (from held-out cases only)   | In-context learning gain |
| `with_case_context`    | Above + case's full prior pair history + initial state | Narrative coherence gain |

**Critical:** few-shot examples must come from cases **outside the evaluation
set** — otherwise we leak answers. Reserve ~10 cases up-front as example-only.

### Models

Minimum 2-model comparison, suggested set:

- **Claude Sonnet 4** — main workhorse, cost/quality balance
- **Claude Opus 4** — ceiling check (is the task solvable by strong LLM?)
- **GPT-4o** or **GPT-5** — cross-family check (is result Claude-specific?)
- *(optional)* **GPT-3.5-turbo** or **Llama 3.1 70B** — weak-model floor

All called via API with temperature 0 for reproducibility.

### Budget planning

808 pairs × ~3k tokens/call × 3 variants × 3 models ≈ **~22k calls, ~65M tokens**.
Mitigations:
- Run `zero_shot` on all models first; only run other variants on top performers
- Cache LLM outputs keyed on `(model, variant, case_id, pair_id)` — reruns are free
- Parallelize with concurrency cap (20-50 concurrent calls)
- Estimate $50-150 total for the full sweep on current Anthropic/OpenAI pricing

### Comparison analysis

Run through same harness, produce parallel CSVs:
```bash
python emsim_eval.py --engine rule_engine:RuleEngine       --export rule.csv
python emsim_eval.py --engine llm_engine:ClaudeSonnet4ZS   --export llm_claude_zs.csv
python emsim_eval.py --engine llm_engine:GPT4oZS           --export llm_gpt4o_zs.csv
```

Then compare:

1. **Headline**: strict pass rate, mean partial score, per-vital MAE side-by-side
2. **Agreement matrix**: for every pair, categorize as
   `{both pass, rule-only pass, llm-only pass, both fail}` → 2×2 confusion
3. **Disagreement analysis**: on pairs where exactly one engine passed, sample
   10 each and manually inspect — what is the LLM catching that rules miss, and
   vice versa?
4. **Per-category winners**: some categories (e.g. rare toxicology) may favor
   LLM; others (stable drift) may favor rules
5. **Cost per correct prediction**: compute `$ per passed pair` for each LLM
   configuration — important for scaling decisions

### Timeline

Fits into the build plan as an independent workstream:

- **Build LLM engine scaffold** — 1 day, parallel to Phase 0-1 of rule engine
- **Run zero-shot sweep** — 1 day wall-clock (mostly API wait time)
- **Analyze + write few-shot prompts** — 1 day
- **Run few-shot / with-context sweeps** — 1 day
- **Write comparison report** — 1 day

Total ~5 days, can start as soon as Phase 0 of rule engine is done (for the
harness and `.step()` signature). Doesn't block rule engine work.

### What "good" looks like

This isn't a competition — we're collecting information. Plausible outcomes:

- **Rule-based wins overall but LLM wins on outliers** → motivates hybrid
- **LLM wins across the board** → reconsider rule-based scope, maybe rules
  become the "deterministic core" and LLM handles extrapolation
- **Rule-based wins on vitals, LLM wins on rare interventions / drugs** →
  justifies the structured approach to physiology; keep LLM for symptom narration
- **Neither hits 50% strict pass** → scenarios may need more constraint than
  either approach can extract; revisit evaluation tolerances or scenario quality

Whichever outcome, the comparison numbers go into `v1_baseline.md` as the
canonical rule-based-vs-LLM reference for the project.

## 12. Exit Criteria for v1

v1 is "done" and ready to wire into the agent loop when:

- Strict pair pass ≥ 50% on the canonical set (excluding known outliers)
- Per-vital MAE: HR ≤ 6, BP_sys ≤ 8, BP_dia ≤ 5, RR ≤ 2, O2Sat ≤ 2.5, T ≤ 0.15
- All-directions-match rate ≥ 70%
- Held-out case set metrics within 5pp of fit-set metrics (no overfitting)
- One documentation file per library (pathology, intervention, drug) listing
  every registered rule with source citation (which scenarios motivated it)
- **LLM baseline sweep completed** (at least zero-shot on 2 models) and
  comparison written up in `v1_baseline.md`

## 13. Tech Debt (tracked)

Items deferred to a later phase cleanup pass:

- `HiddenState` dataclass field defaults currently duplicate `DEFAULT_HEALTHY`
  values. Harmless today (`decode_state` overlays `DEFAULT_HEALTHY` + pathology
  signature on every call, so `HiddenState()` direct instantiation never
  leaks into the hot path) but creates two sources of truth. Phase 5/6
  action: remove scalar defaults; require `HiddenState` to be constructed
  via `decode_state()` or an explicit factory. Add a test asserting no code
  path instantiates `HiddenState()` bare.
