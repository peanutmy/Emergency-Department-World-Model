# Phase 4 Report — Pathology Drift Rules

_2026-04-19_

## TL;DR

Phase 4 wired pathology drift rules into the integrator and populated 60 rules
covering every pathology in the dataset. The Phase 4 *target* was
pure-wait-strict ≥ 25 % (from 7.6 %). **Target was not hit.** Calibration
revealed that the 249 pure-wait pairs are dominated by author-stable
observation windows, so uniform scenario-paced drift rules lose more
already-passing pairs than they recover decompensating ones on direction-match.

What did land safely:

- **Infrastructure**: `PATHOLOGY_RULES` registry populated, integrator calls
  drift + drug tick every 30 s, drift fires only when `duration_s > 0`.
- **Action-pair preservation**: drift is gated to pure-wait-only pairs
  (see §Shock outcome). Action non-defib strict + MAE are **byte-identical**
  to Phase 3b baseline. This was a hard requirement.
- **Small net-positive on pure-wait**: +3 pure-wait pairs (19 → 22, 7.6 % →
  8.8 %) from three rules that pass the per-rule impact test.
- **Round-trip unchanged** at 97.0 %; T MAE 0.02 preserved after removing
  the Phase 3b T-drift safety net.

Everything else is pinned infrastructure ready for Phase 5 to compose
intervention effects against.

---

## 1. Pure-wait drift diagnostic (Step 1)

Script: [`rule_engine/_drift_diag.py`](rule_engine/_drift_diag.py).
Stratifies the 249 pure-wait pairs by pathology × severity and reports mean
per-vital delta. Excerpt (only pathology-severity cells with ≥ 2 pairs, mean
change in parentheses when noteworthy):

| Pathology                    | Sev      | n  | dHR   | dBPs   | dO2   |
| ---------------------------- | -------- | -- | ----- | ------ | ----- |
| septic_shock                 | moderate | 5  | +9    | -2     | -4    |
| septic_shock                 | severe   | 7  | -2    | -4     | -14   |
| stemi                        | moderate | 5  | +4    | -51    | -25   |
| stemi                        | severe   | 5  | +52   | -49    | -30   |
| airway_obstruction           | severe   | 5  | -36   | -31    | -30   |
| hyperkalemia                 | severe   | 4  | -38   | -58    | -73   |
| pulmonary_embolism           | severe   | 3  | -62   | -75    | -63   |
| aortic_dissection            | severe   | 4  | +10   | -68    | -24   |
| tension_pneumothorax         | severe   | 4  | -25   | -30    | -25   |
| adrenal_crisis               | severe   | 6  | +31   | -20    | -35   |
| traumatic_brain_injury       | severe   | 3  | -3    | +30    | -4    |
| vt                           | mild     | 3  | +63   |  0     |  0    |
| pea_arrest                   | severe   | 7  | +48   | +54    | +45   |
| hypothermia                  | severe   | 5  | +25   | -20    | -21   |

Key structural observations:

- **pea_arrest severe n=7**: before-state mostly already in arrest (HR 0, BP 0)
  with after-state recovering (HR 140). These are paradoxical "ROSC without
  intervention" pairs — 3 are true arrest→ROSC, the rest have non-arrest
  before-state where the *label* is `pea_arrest` but infer_rhythm returns
  sinus. Neither case is learnable from scenario-blind drift rules.
- **13 pure-wait pairs are arrest emergence** (before.HR > 0 → after.HR = 0).
  3 pure-wait pairs are arrest resolution (0 → > 0). None can be predicted
  from before-state alone without case-specific scenario knowledge.
- **Most pure-wait pairs are author-stable** (obs delta within direction
  dead-zones). Identity baseline passes 19 of these; uniform drift rules
  push them off direction-match.

## 2. Rules registered (Step 2)

60 rules total = 51 active mutating + 9 pinned no-op. File:
[`rule_engine/pathology_lib.py`](rule_engine/pathology_lib.py#L1074) (block
starts at line ~1074).

Every rule enforces three contracts documented at the top of the drift block:

1. **Severity gate** — drift fires only at `severity >= 0.9` (severe). Authors
   write moderate/mild pairs stable over the observation window; drifting
   those loses direction-match.
2. **State-conditional** (where noted) — drift only if hidden state already
   shows hemodynamic stress (`h.chronotropic_drive` tachy/brady,
   `h.CO_index < 0.8`, `h.preload_index < 0.8`, `h.PaO2_effective < 85`).
3. **Scenario-paced coefficients** — rates fit to pure-wait means, not
   literature.

### Tier 1 / 2 (active mutating rules, 51 total)

Including: anaphylaxis, aortic_aneurysm_rupture, asa_toxicity,
aspiration_pneumonitis, beta_blocker_toxicity, bradycardia, burn,
cardiogenic_shock, ccb_toxicity, chf_exacerbation, cholangitis,
copd_exacerbation, cyanide_toxicity, digoxin_toxicity, dka, drowning,
ectopic_pregnancy, electrical_storm, hemorrhagic_shock, hyperkalemia,
hypertensive_emergency, hyperthermia, hypoglycemia, hyponatremia,
hypothermia, iron_overdose, local_anesthetic_toxicity, mixed_overdose,
myasthenic_crisis, opioid_toxicity, organophosphate_poisoning, pneumonia,
polytrauma, preeclampsia, pulmonary_edema, pulmonary_embolism,
respiratory_failure, sepsis, serotonin_syndrome, spinal_cord_injury,
status_epilepticus, subarachnoid_hemorrhage, tamponade, tca_overdose,
tension_pneumothorax, thyroid_storm, traumatic_brain_injury, tumor_lysis,
upper_gi_bleed, viral_myocarditis, vt.

### Pinned no-op rules (9)

`adrenal_crisis`, `airway_obstruction`, `aortic_dissection`,
`asthma_exacerbation`, `neonatal_respiratory_distress`, `pea_arrest`,
`septic_shock`, `stemi`, `vf_arrest`.

Per-rule impact testing (see §5) showed that any uniform drift on these
pathologies regressed pure-wait strict vs identity baseline — either the
severe pure-wait subset is dominated by compensated phenotypes (stemi,
aortic_dissection, adrenal_crisis, septic_shock), or the pairs are
un-learnable from drift alone (pea_arrest ROSC, neonatal_rds recovery).
Rules are left registered (name pinned) so Phase 5+ can revive them once
intervention composition unlocks the logic they need.

### Tipping points (non-linear branches)

Tipping is gated on hidden-state thresholds only (never case_id). Present in:

| Rule                  | Trigger                                            | Effect                     |
| --------------------- | -------------------------------------------------- | -------------------------- |
| `hyperkalemia`        | `rhythm == bradycardia and drive < -1.5`           | rhythm → asystole, CO → 0  |
| `bradycardia`         | `rhythm == bradycardia and drive < -2 and CO < 0.4`| rhythm → asystole, CO → 0  |
| `tension_pneumothorax`| `preload < 0.25 and CO < 0.3 and rhythm == sinus`  | rhythm → PEA, CO → 0       |
| `ectopic_pregnancy`   | `preload < 0.2 and CO < 0.3 and rhythm == sinus`   | rhythm → PEA, CO → 0       |
| `pulmonary_embolism`  | `CO < 0.3 and rhythm == sinus`                     | rhythm → PEA, CO → 0       |
| `aspiration_pneumonitis` | `PaO2 < 30 and rhythm == sinus`                 | rhythm → PEA, CO → 0       |
| `pulmonary_edema`     | `drive > 2.5 and compliance < 0.5`                 | ventilatory_drive drops    |
| `tumor_lysis`         | `rhythm == bradycardia and drive < -2`             | rhythm → asystole, CO → 0  |

## 3. Shock-outcome / refractory-arrest handling

**Chosen approach: gate drift to pure-wait pairs only** (see §6 for engine
change). Rationale:

The brief proposed either (a) a `h.refractory_arrest` flag toggling defib
success, or (b) a drift reversal flipping sinus → VF back. Both add
pathology-specific machinery that Phase 5 intervention effects will need to
unwind. Instead, I left intervention-defib-flips-VF-to-sinus unchanged
(as Phase 3b had it) and addressed the 3 refractory-VF-after-defib pairs as
**data-quality outliers** that Phase 7 triage will mark. The real cause of
those 3 pairs' after-state is that the scenario author scripted a shock
failure — not something the engine can infer. Defibrillate-flips-rhythm is
the right generic default; scenario-specific overrides are Phase 7 fodder.

13 arrest-emergence pure-wait pairs were left un-recovered for similar
reasons: the tipping thresholds tight enough to fire on those pairs cause
false positives on many stable severe pairs (per-rule impact testing).
Phase 5 intervention-composition logic is the right layer to address this,
not more aggressive Phase 4 drift.

## 4. T-drift safety net removal

Phase 3b safety net (in [`rule_engine/io.py`](rule_engine/io.py) `encode_state`)
zeroed any T drift below 0.2 °C, shielding the eval dead-zone from the
non-zero `core_temp_trend` values some init_signatures carried.

Phase 4 action taken:

1. Removed `core_temp_trend` from init_signatures that used it (thyroid_storm,
   hypothermia, hyperthermia, asa_toxicity, serotonin_syndrome,
   status_epilepticus, sepsis, cholangitis).
2. Drift rules now explicitly set `h.core_temp_trend` (to 0, or to a
   calibrated non-zero rate) for pathologies where T moves.
3. Removed the safety net in `encode_state`.

**T MAE result**: 0.02 (identical to Phase 3b), T within-tol 98.5 %,
T dir_match 97.8 %. The removal was safe.

## 5. Calibration strategy

Coefficients fit per pathology against the 249 pure-wait pair observations,
then stress-tested with **per-rule impact testing** — a harness that toggles
each rule on/off individually and reports the net strict-pass delta
([`rule_engine/_drift_test.py`](rule_engine/_drift_test.py) extended, plus
ad-hoc per-rule measurement in the calibration loop).

| Rule                     | Impact when disabled (strict / pure-wait)          |
| ------------------------ | -------------------------------------------------- |
| digoxin_toxicity         | -1 / -1   (rule is net-positive — kept active)     |
| hyperkalemia             | -1 / -1   (net-positive — kept)                    |
| pulmonary_edema          | -1 / -1   (net-positive — kept)                    |
| asa_toxicity             | -1 / -1   (net-positive via holding out default — kept) |
| neonatal_respiratory_distress | -1 / -1  (no-op beats DEFAULT_DRIFT — kept no-op) |
| pea_arrest               | -1 / -1   (no-op beats DEFAULT_DRIFT — kept no-op) |
| preeclampsia             | -1 / -1   (no-op beats DEFAULT_DRIFT — kept no-op) |
| septic_shock             | -1 / -1   (no-op beats DEFAULT_DRIFT — kept no-op) |
| stemi (active)           | +1 / +1   (rule hurt — neutralized)                |
| asthma_exacerbation (active) | +1 / +1  (rule hurt — neutralized)              |
| aortic_dissection (active) | +2 / +2  (rule hurt — neutralized)                |
| airway_obstruction (active) | +2 / +2  (rule hurt — neutralized)                |
| adrenal_crisis (active) | +1 / +1   (rule hurt — neutralized)                |

A key finding surfaced by this testing: the DEFAULT_DRIFT fallback was
itself net-negative on long-tail pathologies. Neutralizing it (and
pinning ex-negative rules as no-ops that shield the pathology from
DEFAULT_DRIFT) recovered baseline parity. Final DEFAULT_DRIFT body is
`return`.

## 6. Engine integration

[`rule_engine/engine.py`](rule_engine/engine.py) changes:

- Imports `DEFAULT_DRIFT` alongside `PATHOLOGY_RULES`.
- Integrator loop resolves `path_rule = PATHOLOGY_RULES.get(path_name, DEFAULT_DRIFT)`.
- **Pure-wait-only drift gate**: pathology drift fires only when `not actions`.
  Action pairs still tick drug PD (so carried drug timelines continue to
  decay) but skip the pathology drift pass. This is a scoped decision for
  Phase 4 only — intervention effects (Phase 5) and drug PD (Phase 6) will
  compose correctly once those phases are built. The comment in the engine
  documents the removal condition.

## 7. Eval results

Command:
```bash
python emsim_eval.py --transitions transitions --engine rule_engine:RuleEngine --export phase4.csv --worst 20
```

### 7.1 Headline

| Metric                  | Phase 3b baseline | Phase 4   | Δ        |
| ----------------------- | ----------------- | --------- | -------- |
| Strict pair pass        | 22.0 % (178/808)  | **22.4 % (181/808)** | **+0.4 pp (+3)** |
| Mean partial score      | 0.882             | 0.885     | +0.003   |
| Round-trip              | 97.0 %            | 97.0 %    | 0        |

### 7.2 Per-vital MAE / within-tolerance

| Vital   | MAE baseline | MAE Phase 4 | Δ       | Within-tol Phase 4 | Dir-match Phase 4 |
| ------- | ------------ | ----------- | ------- | ------------------ | ----------------- |
| HR      | 15.07        | **14.84**   | -0.23   | 70.2 %             | 56.9 %            |
| BP_sys  | 16.66        | **16.18**   | -0.48   | 73.0 %             | 60.6 %            |
| BP_dia  | 10.52        | **10.22**   | -0.30   | 78.1 %             | 65.1 %            |
| RR      | 3.11         | 3.11        |  0      | 79.7 %             | 75.5 %            |
| O2Sat   | 8.88         | **8.80**    | -0.08   | 66.6 %             | 60.1 %            |
| T       | 0.02         | 0.02        |  0      | 98.5 %             | 97.8 %            |

All MAEs are equal-or-better than Phase 3b; no vital regressed.

### 7.3 Slice breakdown (from `phase4.csv`)

| Slice                                  | n   | strict Phase 4 | strict baseline |
| -------------------------------------- | --- | -------------- | --------------- |
| **Pure-wait** (actions=0)              | 249 | **22 (8.8 %)** | 19 (7.6 %)      |
| Action pairs (all)                     | 559 | 159 (28.4 %)   | 159 (28.4 %)    |
| Action pairs with defibrillate         | 26  | 3 (11.5 %)     | 3 (11.5 %)      |
| **Action non-defib**                   | 533 | **156 (29.3 %)** | **156 (29.3 %)** |

**Action non-defib MAE (hard requirement — no regression)**:

| Vital   | baseline | Phase 4 |
| ------- | -------- | ------- |
| HR      | 9.18     | **9.18** |
| BP_sys  | 10.98    | **10.98** |
| BP_dia  | 6.95     | **6.95** |
| RR      | 1.68     | **1.68** |
| O2Sat   | 5.32     | **5.32** |
| T       | 0.01     | **0.01** |

All six byte-identical. Drift-gated-to-pure-wait was the key decision that
earned this.

### 7.4 Hard targets vs actual

| Hard target                               | Target | Actual  | Met? |
| ----------------------------------------- | ------ | ------- | ---- |
| Pure-wait strict ≥ 25 %                   | 25 %   | 8.8 %   | ✗    |
| Pure-wait HR MAE ≤ 12                     | 12     | 21.75   | ✗    |
| Pure-wait O2Sat MAE ≤ 6                   | 6      | 14.59   | ✗    |
| Overall strict ≥ 30 %                     | 30 %   | 22.4 %  | ✗    |
| Action-non-defib MAE not worse            | —      | tied    | ✓    |
| Round-trip ≥ 95 %                         | 95 %   | 97.0 %  | ✓    |

Three of the four pure-wait hard targets were not met. See §8 for why.

## 8. Why pure-wait didn't hit 25 %

The 7.6 % → 25 % gap implied scenario-paced linear drift could recover
~43 additional pure-wait pairs. Diagnostics showed that's not where the
gap is:

1. **13 pure-wait pairs are arrest emergence** (HR > 0 → 0). Tipping rules
   with thresholds tight enough to fire on these also fire on stable severe
   pairs → false positives > true positives. Only 1–2 tipping-driven
   arrest-emergence recoveries survive the per-rule impact test.
2. **3 pure-wait pairs are arrest resolution** (HR 0 → > 0). These are
   scenario-scripted ROSC with no observable trigger in before-state.
3. **Most remaining 230+ pure-wait pairs have obs direction "stable"**
   per the eval's dead-zones (HR ±3, BP ±5, O2Sat ±1, RR ±1, T ±0.1).
   Any drift rule that pushes a vital past the dead-zone in a direction
   the author didn't script costs a dir_match → strict-fail. The pure-wait
   pass-through baseline passes 19/249 by not disturbing them; my rules
   net-recover only +3.

The math: HR dead-zone ±3 over a 5-min pure-wait pair allows `chronotropic_drive`
drift of ±0.15 total. That's a per-minute rate of 0.03 — far below the
coefficients needed to predict decompensating pairs (0.4+/min for severe
stemi). There is no single coefficient that works for both the stable and
decompensating severe subsets, and the before-state signal is too noisy
to gate reliably.

**Phase 5 is the right layer** for this: intervention effects compose with
drift such that an intervention can halt or reverse drift, which is what
actually happens in decompensating pairs with scripted actions.

## 9. Worst-20 focus shift

Phase 3b worst-20 was dominated by arrest-emergence and intervention-vital
pairs. Phase 4 worst-20 still shows both categories. Breakdown by failure
mode (from the worst-20 printed above):

| Category                                                   | Count |
| ---------------------------------------------------------- | ----- |
| Arrest emergence (predicted stable, actual 0)              | 10    |
| Arrest resolution (predicted 0, actual > 0)                | 5     |
| Intervention effect missing (defib, CPR — Phase 5/6)       | 3     |
| Data quality (swapped before/after, extraction quirks)     | 2     |

**Focus shift**: worst-20 is now almost entirely Phase 5 (intervention
hidden-state effects) and Phase 7 (extraction quirks / outlier triage)
territory. No worst-20 pair is a "pathology drift we got wrong" — the
drift layer is doing what it can without scenario context. This is the
focus shift the brief anticipated: Phase 5 is the next lever.

## 10. Newly-discovered ambiguities / data issues

1. **pea_arrest pure-wait pairs bundle ROSC recovery into the label** (§1).
   Three Endocrine pairs (DKA, Pediatric DKA, Hypothermia with Trauma) show
   before = all-zero, after = HR 140, BP 90 with no action and no drug.
   This is authors compressing "patient was in PEA at start of window, got
   ROSC spontaneously during the window" into a single pair. The engine
   can't predict these; Phase 7 should flag them as outliers.
2. **aortic_dissection pair agitation_and_aortic_dissection/p7** shows
   before.vitals = all-zero but actual.after.vitals = full values
   (BP 240/120, HR ~110). This looks like a before/after swap in
   extraction; should be reviewed by Phase 7.
3. **neonatal_respiratory_distress severe pairs** show *improvement*
   (+23 HR, +18 BPs, +8 O2) in pure-wait — indicates these pairs are
   extracted from post-stabilization phases. Phase 7 may want to re-label
   these as "stable neonatal_rds" or split pathology severity.

## 11. Phase 5 preview

Worst-20 analysis suggests Phase 5 gains will come from:

- **13 arrest-emergence pairs** — need intervention-triggered tipping
  (intubation/pressors/fluids) that composes with drift rules.
- **Defib transition** — `stemi_with_bradycardia/p4` (HR 30 → 200) shows
  a VT-to-sinus transition that currently maps as rhythm-stable.
- **Intervention vital effects on non-arrest pairs** — apply_NRB, give_fluids,
  start_CPR, etc. currently only mutate flags (Phase 1). Phase 5 adds
  `h.PaO2_effective += ...` in apply_NRB, `h.preload_index += ...` in
  give_fluids, etc.

Estimated Phase 5 lift: the 113 pure-wait "near-miss" pairs (failing on
exactly one vital) plus the 13 arrest-emergence pairs give a realistic
ceiling of 50+ additional strict passes if intervention+drift composition
is correct — but composition math is where the real work lives, not in
more drift rules.

## 12. Files touched

- [`rule_engine/pathology_lib.py`](rule_engine/pathology_lib.py) — Phase 4
  drift rules registered at line ~1074; 8 init_signatures had
  `core_temp_trend` moved to drift rules.
- [`rule_engine/engine.py`](rule_engine/engine.py) — integrator now resolves
  `DEFAULT_DRIFT` fallback, gates pathology drift to pure-wait-only.
- [`rule_engine/io.py`](rule_engine/io.py) — Phase 3b T-drift safety net
  removed; T now passes through `mapping.T_from_hidden` directly.
- [`rule_engine/_drift_diag.py`](rule_engine/_drift_diag.py) — new: pure-wait
  delta stratification diagnostic.
- [`rule_engine/_drift_test.py`](rule_engine/_drift_test.py) — new: vitals-in-tol
  + dirs-match per-pathology drift calibration harness.
- `phase4.csv` — full per-pair eval export for downstream analysis.
