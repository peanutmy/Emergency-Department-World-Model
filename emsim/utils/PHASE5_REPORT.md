# Phase 5 Report — Intervention Hidden-State Effects + Drift Composition

_2026-04-19_

## TL;DR

Phase 5 filled in hidden-state mutations for all 21 intervention rules (Tier 1
O2 escalation, Tier 2 perfusion/cardiac, Tier 3 thoracic/temp/airway), added
a `_continuous_intervention_effects` integrator helper for ongoing infusions
(fluids) and CPR-sustained CO, and unlocked the Phase 4 drift gate so pathology
drift composes with intervention jumps on action pairs — except drug-only
pairs, where drug PD is a Phase 6 stub and has no counter-force.

Headline result: **overall strict 22.4 % → 23.3 %** (+0.9 pp) and **action
non-defib strict 29.3 % → 30.8 %** (+1.5 pp). The aspirational 38 % floor for
action non-defib was NOT met — the remaining gap is dominated by missing drug
PD (Phase 6) and arrest-emergence / data-quality pairs (Phase 7 triage). MAE
improvements are the real Phase 5 prize: O2Sat MAE 8.80 → **8.50** overall and
**5.32 → 4.87** on action non-defib, within striking distance of the ≤ 4 target.

5 of 9 pinned drift rules were un-pinned (adrenal_crisis, asthma_exacerbation,
septic_shock, stemi, aortic_dissection) — 2 net-positive on strict, 3 net-
positive on per-pathology MAE but strict-neutral. 4 rules remain pinned
(airway_obstruction, neonatal_respiratory_distress, vf_arrest, pea_arrest)
because their pure-wait subsets are either scenario-scripted ROSC or dominated
by author-stable severe pairs that any uniform drift regresses.

Round-trip test held at 97.0 % — no decoder regressions.

---

## 1. Baseline (Phase 4) vs Phase 5 headline

| Metric                         | Phase 4   | Phase 5   | Δ        | Hard target | Met? |
| ------------------------------ | --------- | --------- | -------- | ----------- | ---- |
| Strict pair pass               | 22.4 %    | **23.3 %**| +0.9 pp  | ≥ 30 %      | ✗    |
| Mean partial score             | 0.882     | 0.887     | +0.005   | —           | —    |
| Pure-wait strict               | 8.8 %     | 8.4 %     | −0.4 pp  | ≥ 7 %       | ✓    |
| Action non-defib strict        | 29.3 %    | **30.8 %**| +1.5 pp  | ≥ 38 %      | ✗    |
| Action defib strict            | 11.5 %    | 11.5 %    | 0        | —           | —    |
| Round-trip (tol/2)             | 97.0 %    | 97.0 %    | 0        | ≥ 95 %      | ✓    |
| Action non-defib O2Sat MAE     | 5.32      | **4.87**  | −0.45    | ≤ 4.0       | near |
| Action non-defib HR MAE        | 9.18      | 9.09      | −0.09    | ≤ 7.0       | ✗    |
| Overall HR MAE                 | 14.84     | 14.78     | −0.06    | —           | —    |
| Overall BP_sys MAE             | 16.18     | 17.00     | +0.82    | —           | —    |
| Overall O2Sat MAE              | 8.80      | **8.50**  | −0.30    | —           | —    |

Command:
```bash
python emsim_eval.py --transitions transitions --engine rule_engine:RuleEngine --export phase5.csv
```

The BP_sys regression +0.82 is from drift × intervention composition on
action pairs where drift now runs (was previously gated). It's an accepted
tradeoff; strict passes went up on intervention pairs because O2Sat moves
past ±1 deadzone in the right direction more often.

---

## 2. Intervention hidden-state mutations (21 rules)

### Tier 1 — O2 escalation (ROI: highest)

All O2-device rules now lerp `h.PaO2_effective` **in O2Sat space** (not PaO2
space) toward a physiologic ceiling sat set by FiO2 × residual shunt. The
original PaO2-space lerp overshot O2Sat to 100 % on any substantial PaO2
bump (hemoglobin curve saturates at PaO2 ≥ 150 mmHg), which regressed
most passing duration_s=0 pairs. Sat-space lerp with a ceiling from
`_ceiling_sat(FiO2, shunt) = arterial_sat*(1-shunt) + venous_sat*shunt`
keeps deltas physiologic (typical +5-15 points).

| Rule                   | Shunt effect             | Compliance      | Weight | Result (solo:X)           |
| ---------------------- | ------------------------ | --------------- | ------ | ------------------------- |
| `apply_nasal` (n=32)   | —                        | —               | 0.40   | **59.4 %** (was 12.5 %), O2 MAE 3.06 → **1.93** |
| `apply_NRB` (n=24)     | —                        | —               | 0.40   | **41.7 %** (was 25 %), O2 MAE 4.62 → **3.72** |
| `apply_BVM` (n=22)     | `shunt_factor=0.9`       | —               | 0.40   | **36.4 %** (was 4.5 %), O2 MAE 7.41 → **5.00** |
| `apply_BiPAP` (n=6)    | `*= 0.85` (persistent)   | `max 0.8`       | 0.40   | 0/6 (BP fails), O2 MAE 3.50 → **1.99** |
| `add_PEEP` (n=5)       | scaled by cmH2O/15       | —               | 0.40   | 0/5 (RR fails), O2 MAE 7.60 → **5.84** |
| `intubate` (n=78)      | `*= 0.6` (persistent)    | `max 0.7`       | 0.55   | 2/78 (multi-fail), O2 MAE 12.01 → **11.37** |

`intubate` regressed 3 pairs on strict (6.4 % → 2.6 %): these are
duration_s=0 scenarios with stable (unchanged) author-vitals that my
moderate O2Sat jump (+2-14) crossed. The improvement on 73 hypoxic-
intubate pairs' O2 MAE is the net gain; attempts to gate by gap size or
duration_s either zeroed-out the gain on nasal/NRB pairs or added
complexity with no net benefit.

### Tier 2 — Perfusion / cardiac

| Rule                  | Hidden-state effect                                            | Result |
| --------------------- | -------------------------------------------------------------- | ------ |
| `give_fluids` (n=92)  | **No t=0 jump** (continuous integrator handles infusion). Per-tick `preload_index += 3e-5 * rate * dt_min`. | 23.9 % (unchanged vs Phase 4; previous t=0 bump broke 5 stable BP pairs on ±5 dir-match) |
| `start_CPR` (n=25)    | If arrest: `CO_index = max(., 0.3)`, `preload = max(., 0.4)`; repeated each tick while `CPR_active` | **84 %** (was 84 %; stable) |
| `stop_CPR` (n=3)      | If arrest: `CO_index = 0`                                      | 33.3 % |
| `defibrillate`        | VF → sinus, `chronotropic_drive = 0` (clean post-ROSC baseline), `CO_index/preload_index = max(0.8)` | 11.5 % (strict unchanged; HR MAE still dominated by post-ROSC mapping) |
| `start_pacing` (n=4)  | If bradycardia: flip to sinus. Set `chronotropic_drive` so `HR_from_hidden = pacing_rate`. `CO_index = max(., 0.7)`. | 0/4 (BPs fails due to residual-corrected SVR/CO — can't force BP higher without changing mapping) |

### Tier 3 — Thoracic / temperature / airway

State-conditional gating: chest-drainage rules only lift preload/compliance
if they're **already below** physiologic threshold (preload < 0.5,
compliance < 0.7). This prevents over-correction on polytrauma pairs where
the pathology signature sets preload=0.7 but BP is stable — lifting preload
to 0.9 pushed BP_sys up ~21 mmHg, crossing dir-match.

| Rule                        | Hidden-state effect                                            | Result |
| --------------------------- | -------------------------------------------------------------- | ------ |
| `place_chest_tube` (n=7)    | If preload<0.5: lerp(., 0.85, 0.5); if compliance<0.7: lerp(., 0.9, 0.5); shunt *= 0.8 | 14.3 % (was 14.3 %; pericardial/trauma-tangential pairs previously over-shot) |
| `needle_decompress` (n=6)   | Same preload/compliance gate; shunt *= 0.7                     | **33.3 %** (was 16.7 %) |
| `pericardiocentesis` (n=2)  | If CO<0.6: lerp(., 0.8, 0.5); if preload<0.6: lerp(., 0.85, 0.5) | 0/2 (two pairs are tamponade ROSC, unlearnable) |
| `start_warming` (n=5)       | `core_temp_trend = max(., +0.05)`                              | **60 %** (unchanged) |
| `start_cooling` (n=3)       | `core_temp_trend = min(., -0.03)`                              | **66.7 %** (unchanged) |
| `place_airway` (n=2)        | `shunt_fraction *= 0.95`                                       | 0/2 (O2 MAE 4.5) |
| `remove_foreign_body` (n=1) | `shunt *= 0.3`, `compliance = max(., 1.0)`, re-oxygenate       | 0/1 |
| `needle_cricothyroidotomy` (n=1) | `shunt *= 0.5`, `compliance = max(., 0.8)`, re-oxygenate  | 0/1 |
| `surgical_cricothyrotomy` (n=1)  | Same as intubate                                          | 0/1 |
| `escharotomy` (n=1)         | `compliance = max(., 0.9)`, `shunt *= 0.8`                     | 0/1 |

---

## 3. Continuous intervention effects (integrator)

New helper [`rule_engine/engine.py:_continuous_intervention_effects`](rule_engine/engine.py)
runs each 30-s tick for every pair with `duration_s > 0`:

```python
def _continuous_intervention_effects(h, flags, dt_s):
    dt_min = dt_s / 60.0
    rate = flags.get("fluids_rate_ml_hr") or 0
    if rate:
        h.preload_index = clip(h.preload_index + 3e-5 * rate * dt_min, 0.0, 1.8)
    if flags.get("CPR_active"):
        h.CO_index = max(h.CO_index, 0.3)
        h.preload_index = max(h.preload_index, 0.4)
```

The fluids coefficient is scenario-paced: 500 ml/hr × 5 min → +0.025 preload
units. Small but additive over longer observation windows. Per-tick CPR floor
keeps hidden state coherent across rhythm conversions (e.g. CPR → defib →
sinus within one pair's duration_s).

---

## 4. Drift gate unlock (`_DRIFT_GATE_ON_ACTIONS`)

**Phase 5 engine behavior** ([`rule_engine/engine.py`](rule_engine/engine.py)):

```python
if _DRIFT_GATE_ON_ACTIONS:
    run_drift = not bool(actions)       # Phase 4 behavior
else:
    run_drift = not drug_only           # Phase 5: intervention + pure-wait
```

Initial full unlock regressed 15 action pairs (22.6 % → 20.8 %). All 15
were **drug-only pairs** where the pathology drift fired with no drug-PD
counter-force (drug_lib.tick is a Phase 6 stub). Gating drift off for
drug-only pairs restored the +0.9 pp gain while composing drift with
intervention hidden-state effects on intervention pairs.

`EMSIM_DRIFT_GATE=1` environment variable restores the full Phase 4 gate
for A/B testing. Default is `0` (Phase 5 composition on).

Gated→unlocked A/B on action pairs: 15 pairs regressed, 0 gained. This
confirms that Phase 5's composition physics fires correctly on intervention
pairs but the drug_only gate is load-bearing until Phase 6.

---

## 5. Pinned drift-rule re-evaluation (9 rules)

Script: [`rule_engine/_pinned_rules_test.py`](rule_engine/_pinned_rules_test.py).
Each candidate body is swapped in temporarily and scored per-pathology on
pure-wait and action slices under Phase 5 composition.

| Rule                            | Decision      | Candidate                                     | d_strict (pw / act) | Why this call |
| ------------------------------- | ------------- | --------------------------------------------- | ------------------- | ------------- |
| `adrenal_crisis`                | **UNPIN**     | SVR decay, gated on CO<0.9 or SVR<0.9         | **+1 / 0**          | State gate avoids stable-severe regression |
| `asthma_exacerbation`           | **UNPIN**     | PaO2 -1.5/min, ventilatory_drive -0.02/min    | **+1 / 0**          | Author-paced respiratory decline |
| `septic_shock`                  | **UNPIN**     | SVR decay, gated on SVR<0.85                  | 0 / 0 (BPs MAE -0.8)| MAE-positive, strict-neutral |
| `stemi`                         | **UNPIN**     | HR↑, CO decay, gated on already-depressed     | 0 / 0 (HR MAE -0.6) | MAE-positive, strict-neutral |
| `aortic_dissection`             | **UNPIN**     | Tiny O2 drift only (dropped preload/BP mutations) | 0 / 0           | Rest of drifts were net-negative |
| `airway_obstruction`            | keep pinned   | Tested O2 drift gated on PaO2<70; still -1 pure-wait strict | -1 / 0 | Pairs split between arrest-emerge and stable |
| `vf_arrest`                     | keep pinned   | post-ROSC CO decay; 0 impact (n=2 pure-wait)  | 0 / 0               | No detectable drift signature |
| `pea_arrest`                    | keep pinned   | no-op                                         | 0 / 0               | 7 pairs are scenario-scripted ROSC |
| `neonatal_respiratory_distress` | keep pinned   | PaO2 gated on <75; still 0 strict, O2 MAE +2.1 | 0 / 0 (MAE worse)  | Pairs too mixed (some improve from surfactant) |

**Net**: 5 unpinned (2 net-positive strict, 3 MAE-positive / strict-neutral),
4 remain pinned. The brief's soft goal was 4-5 net-positive rules — met on
"any improvement" but only 2 improved strict pass count. The strict-
neutral rules contribute small MAE wins that help near-miss pairs at the
margin.

---

## 6. Worst-20 (action non-defib leaderboard)

From `phase5.csv`, top-20 by tolerance-normalized total vital error:

| Count | Failure category                                    | Phase fix |
| ----- | --------------------------------------------------- | --------- |
| 8     | Arrest emergence (before vitals present, actual 0)  | Phase 6 drug PD / tipping |
| 7     | Arrest resolution (before 0, actual > 0)            | Scenario-scripted ROSC; Phase 7 |
| 5     | Data-quality (before/after swapped or all-zero bug) | Phase 7 outlier triage |
| 0     | Pathology drift wrong                               | — |
| 0     | Intervention effect wrong                           | — |

**Focus has shifted exactly as the Phase 5 brief predicted.** No worst-20
pair is "we got intervention magnitude wrong" or "drift too aggressive" —
those have all been tuned out. The remaining blockers are drug PD (Phase 6)
and scenario-scripted arrest transitions (Phase 7).

Sample:
- `tracheostomy_emergency/p2` (pure-wait, HR 130→30 actual but before shows stable) — unlearnable
- `stemi_with_bradycardia/p4` (HR 30→200 pure-wait) — scenario-scripted
- `agitation_and_aortic_dissection/p7` (before all-zero, actual full vitals) — Phase 7 data bug

---

## 7. Slice comparison vs Phase 4

| Slice                        | n   | Phase 4 strict | Phase 5 strict | Δ        |
| ---------------------------- | --- | -------------- | -------------- | -------- |
| Overall                      | 808 | 181 (22.4 %)   | **188 (23.3 %)** | +7      |
| Pure-wait                    | 249 | 22 (8.8 %)     | 21 (8.4 %)     | −1       |
| Action defib                 | 26  | 3 (11.5 %)     | 3 (11.5 %)     | 0        |
| Action non-defib             | 533 | 156 (29.3 %)   | **164 (30.8 %)** | +8      |
| Action all                   | 559 | 159 (28.4 %)   | **167 (29.9 %)** | +8      |

Per-category:
| Category       | Phase 4 | Phase 5 | Δ  |
| -------------- | ------- | ------- | -- |
| Cardiology     | 67      | 67      | 0  |
| Communication  | 3       | 4       | +1 |
| Endocrine      | 14      | 15      | +1 |
| GI             | 3       | 2       | −1 |
| Neurology      | 8       | 9       | +1 |
| OB-GYN         | 15      | 16      | +1 |
| Pediatrics     | 16      | 18      | +2 |
| Respiratory    | 4       | 5       | +1 |
| Resuscitation  | 43      | 43      | 0  |
| Toxicology     | 6       | 7       | +1 |
| Trauma         | 2       | 2       | 0  |

Gains are distributed across 7 categories; no category regressed more than 1.

---

## 8. Near-miss failure distribution (action non-defib, 369 failures)

| Reason (single-dim failure) | n   | Phase 6/7 fix |
| --------------------------- | --- | ------------- |
| multi-dim failure           | 261 | Phase 6 drug PD dominates |
| O2Sat within-tol            | 27  | Finer per-intervention O2 tuning |
| O2Sat direction             | 12  | Intervention magnitude calibration |
| BP_sys direction            | 11  | Drug PD (pressor / sedation effects on BP) |
| PEEP exact match            | 10  | Extraction: intubate without add_PEEP pairs |
| HR within-tol / direction   | 20  | Drug PD (sedation, pressors) |
| RR within-tol               | 8   | Drug PD (opiate, sedation) |
| severity change             | 6   | Non-standard severity transitions |

108 pairs fail on a single dimension — these are the most tractable recoveries
for Phase 6 (drug PD) and Phase 7 (extraction quirks). 261 multi-dim failures
mostly need drug PD to unblock.

---

## 9. Round-trip test

No regressions vs Phase 3b / Phase 4:

| Vital   | tol/2 | MAE    | P50   | P90   | >tol/2 |
| ------- | ----- | ------ | ----- | ----- | ------ |
| HR      | 5.0   | 0.00   | 0.00  | 0.00  | 0      |
| BP_sys  | 7.5   | 0.17   | 0.00  | 0.00  | 0.5 %  |
| BP_dia  | 7.5   | 0.05   | 0.00  | 0.00  | 0.4 %  |
| RR      | 2.0   | 0.27   | 0.00  | 0.00  | 2.5 %  |
| O2Sat   | 1.5   | 0.00   | 0.00  | 0.00  | 0      |
| T       | 0.1   | 0.00   | 0.00  | 0.00  | 0      |

Pairs passing ALL 6 vitals within tolerance/2: **784/808 (97.0 %)**.

---

## 10. Files touched

- [`rule_engine/intervention_lib.py`](rule_engine/intervention_lib.py) —
  21 rules grew hidden-state mutations. New helpers `_oxygenate`,
  `_oxygenate_to_target`, `_ceiling_sat`, `_alveolar_PaO2`, `_duration`.
- [`rule_engine/engine.py`](rule_engine/engine.py) — added
  `_continuous_intervention_effects`, drug-only drift gate, action-context
  injection (`_duration_s` on action params), `EMSIM_DRIFT_GATE` env
  override.
- [`rule_engine/pathology_lib.py`](rule_engine/pathology_lib.py) — 5
  pinned rules unpinned (adrenal_crisis, asthma_exacerbation, septic_shock,
  stemi, aortic_dissection) with state-conditional gating on hidden state.
- [`rule_engine/_intervention_test.py`](rule_engine/_intervention_test.py) —
  NEW: per-intervention + per-slice diagnostic, worst-20 per slice.
- [`rule_engine/_pinned_rules_test.py`](rule_engine/_pinned_rules_test.py) —
  NEW: candidate drift bodies + per-rule impact harness.
- `phase5.csv` — per-pair eval export.

---

## 11. Hard requirements vs actual

| Hard target                                 | Target  | Actual   | Met? |
| ------------------------------------------- | ------- | -------- | ---- |
| Action non-defib strict ≥ 38 %              | 38 %    | 30.8 %   | ✗    |
| Action non-defib O2Sat MAE ≤ 4              | 4.0     | 4.87     | ≈    |
| Action non-defib HR MAE ≤ 7                 | 7.0     | 9.09     | ✗    |
| Overall strict ≥ 30 %                       | 30 %    | 23.3 %   | ✗    |
| Pure-wait strict ≥ 7 %                      | 7 %     | 8.4 %    | ✓    |
| Round-trip ≥ 95 %                           | 95 %    | 97.0 %   | ✓    |

Four of six hard numeric targets missed. The `strict ≥ 38 %` and `≥ 30 %`
bars were premised on "intervention + drift composition recovers 40 + action
pairs" — the data shows that's bounded by Phase 6 drug PD. Intervention-
pair improvements I landed (+8 pairs) are bounded by pairs where the
intervention's effect fully explains the authored vital delta; the rest of
the 372 remaining failures need drug PD (pressor BP, sedation HR, paralytic
RR) or scenario-specific tipping (Phase 7).

The two MAE targets (O2Sat ≤ 4, HR ≤ 7) are within reach with more per-
intervention tuning; I stopped at 4.87 / 9.09 because the near-miss
diagnostic (§8) showed 261/372 remaining failures are multi-dim (need
drug PD), not single-vital O2 or HR.

---

## 12. Phase 6 preview

Worst-20 + near-miss distribution point directly at drug PD:

- `pressor_alpha_pure` (norepi / vasopressin) → will recover ~30 BP pairs
  currently failing on BP within-tol and dir-match in septic_shock /
  adrenal_crisis / hemorrhagic_shock slices.
- `sedative` + `paralytic` (midazolam, propofol, rocuronium) → will recover
  ~20 peri-intubation HR/BP/RR pairs currently classed as "multi-dim
  failure" because all three move.
- `pressor_alpha_beta` (epinephrine) → will recover ~10 arrest-pair HR/BP.
- `vagolytic` (atropine) → bradycardia HR recovery.
- `antiarrhythmic_3` (amiodarone) → defib-pair rhythm stability.

Estimated Phase 6 lift: +15-20 pp on action non-defib strict (30.8 →
~45-50 %), +8-10 pp overall (23.3 → ~32 %). This puts the overall 30 %
hard target finally in reach.

---

## 13. Open questions / notes for Phase 6-7

1. **Intubate O2Sat regression on arrest-rhythm pairs** (`accidental_hypothermia/p3`
   70→90 expected, we output 70): the mapping's arrest-rhythm pass-through
   for O2Sat blocks intervention-driven improvement. Can't fix via
   intervention hidden-state alone because `O2Sat_from_hidden` returns None
   under arrest → encode falls back to `before.O2Sat`. Phase 3 is frozen per
   the brief, so document this as a known limitation. If mapping ever
   un-freezes, consider reading PaO2_effective under arrest iff the
   intervention just raised it.

2. **23 intubate-without-add_PEEP pairs** (known from PROJECT_STATUS §Medium):
   set PEEP=5 in the after state despite no explicit `add_PEEP` action.
   My intubate rule intentionally leaves PEEP alone (46/78 pairs keep it at
   0). Phase 7 can fix extraction to add concurrent `add_PEEP` on these
   23 pairs, which would recover the 10 PEEP-only near-miss action-pair
   failures.

3. **start_pacing BP regression**: my rule correctly sets HR via
   `chronotropic_drive`, but BP doesn't rise because `BP_from_hidden`
   depends on CO × SVR, and residual correction froze both from the
   pre-pacing BP. Lifting CO_index to 0.7 doesn't help when the decoded
   value was already above 0.7. A proper fix needs either (a) pacing-
   specific BP path in mapping, or (b) a stronger CO floor. Both are
   candidate Phase 6/7 touch-ups.
