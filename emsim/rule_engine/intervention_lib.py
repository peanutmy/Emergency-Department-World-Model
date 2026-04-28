"""
Intervention rule registry.

Each rule has signature `fn(h, flags, params)`:
- `h`      : HiddenState being mutated
- `flags`  : schema `interventions` dict (mutable)
- `params` : action dict from the pair's `actions[]` (carries optional `value`,
             `dose`, `unit`, `route` per schema.json#/definitions/action)

Rules fire once at t=0 as discrete jumps (ENGINE_DESIGN.md §5, §6). After the
intervention jump, the integrator applies pathology drift and drug PD on top
of the mutated hidden state.

Phase 5 coverage: flag mutations **plus** hidden-state mutations so vitals
mapping produces the post-intervention vitals. The hidden-state deltas are
lerp-style (current value toward a physiologic target) so patients with more
severe pathology (higher shunt, lower compliance) respond less than healthy
patients — exactly the clinical pattern.

Alveolar PaO2 approximation (mmHg) used by O2-device rules:
    alveolar(FiO2) ≈ 100 + 600 * (FiO2 - 0.21)
    nasal 0.30 → 154 ; BiPAP 0.50 → 274 ; NRB/BVM/vent 1.0 → 574
PaO2_effective is nudged toward alveolar by weight (1 - shunt_fraction):
healthy (shunt=0) closes the full gap; ARDS (shunt=0.5) closes half.
"""
from __future__ import annotations

from typing import Any, Callable

from .hidden_state import ARREST_RHYTHMS, HiddenState
from .mapping import clip, hemoglobin_curve, inverse_hemoglobin_curve, lerp


# ======================================================================
# Physiology helpers (local to intervention rules)
# ======================================================================
#
# Design choice: oxygenation interventions lerp in **O2Sat space** rather
# than PaO2 space. Rationale: the hemoglobin dissociation curve saturates
# above PaO2≈150 mmHg, so any physiologically plausible PaO2 rise from a
# low baseline slams O2Sat to 100% — far above what the dataset actually
# records for duration_s=0 intubation / NRB pairs (typical dataset deltas
# are +5 to +15 O2Sat points at t=0, not "jump to 100%"). Lerping in sat
# space lets us target a physiologic "ceiling" (set by residual shunt
# mixing venous and arterial blood) at a bounded weight.


def _oxygenate_to_target(h: HiddenState, ceiling_sat: float,
                         weight: float = 0.4) -> None:
    """Pull observed O2Sat toward `ceiling_sat` at `weight` fraction of the gap.

    - Maps current PaO2_effective → O2Sat via the dissociation curve.
    - Lerps toward `ceiling_sat` by `weight`.
    - Maps back via the inverse curve, clipped to [0, 600].
    - Never decreases O2Sat.

    `ceiling_sat` should be the equilibrium saturation the patient can
    achieve under the new FiO2 × residual-shunt combo. Caller computes it
    from the device physiology.
    """
    if ceiling_sat <= 0:
        return
    current_sat = hemoglobin_curve(h.PaO2_effective)
    if current_sat >= ceiling_sat:
        return
    new_sat = lerp(current_sat, ceiling_sat, weight)
    new_sat = max(new_sat, current_sat)
    h.PaO2_effective = clip(
        inverse_hemoglobin_curve(new_sat),
        0.0, 600.0,
    )


def _ceiling_sat(FiO2: float, shunt: float) -> float:
    """Equilibrium O2Sat given FiO2 × residual shunt.

    Two-compartment model: arterial blood is a mix of `(1 - shunt)` well-
    oxygenated blood (saturating close to 100%) and `shunt` venous blood
    (O2Sat ~70-75% at PvO2 ~ 40 mmHg). Low FiO2 also caps the arterial
    side below 100%. Keeps the math simple and avoids hemoglobin-curve
    overshoot.
    """
    # Arterial ceiling under this FiO2:
    #   room air (0.21) → ≈96%, nasal (0.3) → ≈98%, NRB+ (0.5+) → 100%.
    if FiO2 <= 0.21:
        arterial_sat = 96.0
    elif FiO2 <= 0.3:
        arterial_sat = 98.0
    elif FiO2 <= 0.5:
        arterial_sat = 99.5
    else:
        arterial_sat = 100.0
    shunt = clip(shunt, 0.0, 0.8)
    venous_sat = 72.0
    return arterial_sat * (1.0 - shunt) + venous_sat * shunt


def _oxygenate(h: HiddenState, FiO2: float,
               shunt_factor: float = 1.0,
               weight: float = 0.4,
               duration_s: float = 0.0) -> None:
    """Standard oxygenation jump used by O2-device rules.

    `shunt_factor < 1.0` transiently reduces the patient's effective shunt
    for *this* step, modelling PEEP / PPV recruitment on top of a raw FiO2
    jump. `weight` is the fraction of the sat gap closed at t=0 (typical
    scenario deltas calibrated around 0.35-0.55).

    `duration_s` is accepted for future use (e.g. per-tick ventilation
    equilibration) but currently ignored — intervention oxygenation fires
    as a single t=0 jump. Gap-attenuation was tried and lost more pairs
    on direction-match than it saved; simple single-weight lerp wins.
    """
    eff_shunt = clip(h.shunt_fraction * shunt_factor, 0.0, 0.8)
    ceiling = _ceiling_sat(FiO2, eff_shunt)
    _oxygenate_to_target(h, ceiling, weight=weight)


def _duration(params) -> float:
    """Extract duration_s from the action context dict (engine injects this)."""
    return float(params.get("_duration_s", 0.0) or 0.0)


INTERVENTION_RULES: dict[str, Callable[[HiddenState, dict[str, Any], dict[str, Any]], None]] = {}


def intervention_rule(name: str):
    """Decorator: register an intervention rule for `name`."""
    def register(fn: Callable[[HiddenState, dict[str, Any], dict[str, Any]], None]):
        INTERVENTION_RULES[name] = fn
        return fn
    return register


# ======================================================================
# Airway & ventilation (8)
# ======================================================================
# FiO2 defaults — EXTRACTION_GUIDE.md "FiO2 defaults by device" table.
# Phase 5: each rule nudges PaO2_effective toward the alveolar target
# for its FiO2, attenuated by current shunt_fraction. ETT and BiPAP
# additionally recruit alveoli (shunt_fraction reduced persistently).

@intervention_rule("apply_nasal")
def _apply_nasal(h, flags, params):
    flags["O2_device"] = "nasal"
    flags["FiO2"] = 0.3
    _oxygenate(h, 0.3, duration_s=_duration(params))


@intervention_rule("apply_NRB")
def _apply_NRB(h, flags, params):
    flags["O2_device"] = "NRB"
    flags["FiO2"] = 1.0
    _oxygenate(h, 1.0, duration_s=_duration(params))


@intervention_rule("apply_BVM")
def _apply_BVM(h, flags, params):
    flags["O2_device"] = "BVM"
    flags["FiO2"] = 1.0
    # Positive-pressure bagging provides mild recruitment beyond passive NRB.
    _oxygenate(h, 1.0, shunt_factor=0.9, duration_s=_duration(params))


@intervention_rule("apply_BiPAP")
def _apply_BiPAP(h, flags, params):
    flags["O2_device"] = "BiPAP"
    flags["FiO2"] = 0.5
    # Non-invasive PPV recruits alveoli → persistent shunt reduction + this-step
    # oxygenation at FiO2=0.5.
    h.shunt_fraction = clip(h.shunt_fraction * 0.85, 0.0, 0.8)
    h.compliance_index = max(h.compliance_index, 0.8)
    _oxygenate(h, 0.5, duration_s=_duration(params))


@intervention_rule("add_PEEP")
def _add_PEEP(h, flags, params):
    # `value` carries cmH2O; default 5 if scenario omitted it.
    value = params.get("value", 5) or 5
    flags["PEEP"] = value
    # PEEP recruits alveoli → shunt reduction scaled by cmH2O (0..15).
    # 5 cmH2O → ×0.83 (slight), 15 cmH2O → ×0.5 (maximal benefit).
    frac = clip(value / 15.0, 0.0, 1.0)
    h.shunt_fraction = clip(h.shunt_fraction * lerp(1.0, 0.5, frac),
                            0.0, 0.8)
    # Re-oxygenate at whatever current FiO2 is (PEEP without O2 increase still
    # improves sat via the reduced shunt).
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))


@intervention_rule("intubate")
def _intubate(h, flags, params):
    # Data-derived (see calibration note): vent_rate 12 and vent_TV_ml 500
    # are the dominant defaults; FiO2 goes to 1.0 in 77/78 dataset pairs;
    # airway (adjunct) is NOT set — EXTRACTION_GUIDE §8 reserves `airway`
    # for OPA/NPA. PEEP is deliberately left alone (46/78 pairs keep it at 0).
    flags["intubated"] = True
    flags["O2_device"] = "vent"
    flags["FiO2"] = 1.0
    if flags.get("vent_rate") is None:
        flags["vent_rate"] = 12
    if flags.get("vent_TV_ml") is None:
        flags["vent_TV_ml"] = 500
    # Mechanical ventilation recruits alveoli → persistent shunt improvement,
    # compliance floor; then lerp PaO2 toward alveolar at FiO2=1.0.
    # Higher weight (0.55) than passive O2 devices — PPV equilibrates faster
    # and dataset deltas on hypoxic pre-intubation pairs are larger (+15-25
    # vs +5-10 for nasal/NRB).
    h.shunt_fraction = clip(h.shunt_fraction * 0.6, 0.0, 0.8)
    h.compliance_index = max(h.compliance_index, 0.7)
    _oxygenate(h, 1.0, weight=0.55, duration_s=_duration(params))
    # NOTE: we deliberately do NOT model sedation-driven HR drop here
    # (per design decision — requires explicit drug action in scenario)


@intervention_rule("place_airway")
def _place_airway(h, flags, params):
    # OPA / NPA adjunct (not ETT). EXTRACTION_GUIDE §8: airway flag only.
    # Minor airway patency improvement (small shunt reduction).
    flags["airway"] = True
    h.shunt_fraction = clip(h.shunt_fraction * 0.95, 0.0, 0.8)


@intervention_rule("remove_foreign_body")
def _remove_foreign_body(h, flags, params):
    # Relieving foreign-body airway obstruction: large reduction in
    # effective shunt + compliance restored. Raw FiO2 unchanged (O2 device
    # state keeps whatever the scenario had it at).
    h.shunt_fraction = clip(h.shunt_fraction * 0.3, 0.0, 0.8)
    h.compliance_index = max(h.compliance_index, 1.0)
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))


# ======================================================================
# Fluids (1)
# ======================================================================

@intervention_rule("give_fluids")
def _give_fluids(h, flags, params):
    # `value` = ml/hr (schema action.value). fluid_type defaults to crystalloid
    # (EXTRACTION_GUIDE §8) but only when unset — don't clobber a prior colloid/blood.
    rate = params.get("value")
    if rate is not None:
        flags["fluids_rate_ml_hr"] = rate
    if flags.get("fluid_type") is None:
        flags["fluid_type"] = "crystalloid"
    # No t=0 preload bump — `give_fluids` is a continuous infusion. For
    # duration_s=0 pairs the scenario is just marking "IV started" and
    # should show no vital change; the integrator's continuous-infusion
    # helper (_continuous_intervention_effects) accumulates preload for
    # duration_s > 0 pairs.


# ======================================================================
# Arrest / cardiac (4)
# ======================================================================

@intervention_rule("start_CPR")
def _start_CPR(h, flags, params):
    flags["CPR_active"] = True
    # Chest compressions produce ~25-30% of baseline CO. This only surfaces
    # in the mapping if rhythm ever exits arrest (e.g. via defib during the
    # same pair) — under persistent arrest, BP_from_hidden still returns 0.
    # Setting it anyway keeps the hidden state physiologically coherent.
    if h.rhythm in ARREST_RHYTHMS:
        h.CO_index = max(h.CO_index, 0.3)
        h.preload_index = max(h.preload_index, 0.4)


@intervention_rule("stop_CPR")
def _stop_CPR(h, flags, params):
    flags["CPR_active"] = False
    # Persistent arrest → circulation stops; sinus after ROSC keeps whatever
    # CO_index rule composition produced.
    if h.rhythm in ARREST_RHYTHMS:
        h.CO_index = 0.0


@intervention_rule("defibrillate")
def _defibrillate(h, flags, params):
    # Records last joule delivered; CPR toggle is scenario-specific.
    flags["defib_last_J"] = params.get("value", 200)
    # Successful defibrillation of VF → sinus. asystole / PEA don't respond
    # to defib (non-shockable rhythms, clinical consensus). VT conversion
    # is left as data-driven: Phase 2 heuristic rarely labels VT so we
    # skip it here to avoid false ROSC on persistent tachy rhythms.
    if h.rhythm == "VF":
        h.rhythm = "sinus"
        # Post-ROSC: reset chronotropic_drive so HR_from_hidden returns
        # a clean sinus baseline (~80) instead of the pre-arrest signature
        # value (~2.5 SD for vf_arrest → 130 bpm, too high).
        h.chronotropic_drive = 0.0
        # Give the patient a starting CO (pre-defib arrest may have driven
        # it to 0 via stop_CPR or pathology rule).
        h.CO_index = max(h.CO_index, 0.8)
        h.preload_index = max(h.preload_index, 0.8)


@intervention_rule("start_pacing")
def _start_pacing(h, flags, params):
    flags["pacing_active"] = True
    rate = params.get("value")
    if rate is not None:
        flags["pacing_rate"] = rate
    # Transvenous / transcutaneous pacing takes over rate control. We can't
    # route through HR_from_hidden (rhythm-switch would need a new enum);
    # instead, if the patient is bradycardic, lift rhythm to sinus and
    # set chronotropic_drive so HR_from_hidden reproduces `pacing_rate`.
    target = float(rate) if rate is not None else 80.0
    if h.rhythm == "bradycardia":
        h.rhythm = "sinus"
    # Solve: HR = HR_SINUS_BASELINE (80) + HR_SINUS_GAIN (20) * drive
    h.chronotropic_drive = clip((target - 80.0) / 20.0, -4.0, 10.0)
    # Pacing also restores some CO by correcting the rate-driven deficit.
    h.CO_index = max(h.CO_index, 0.7)


# ======================================================================
# Chest / thoracic (3)
# ======================================================================

@intervention_rule("place_chest_tube")
def _place_chest_tube(h, flags, params):
    flags["chest_tube"] = True
    # Drainage relieves intrathoracic compression → preload and compliance
    # recover. Gated on hidden-state signatures of tension physiology
    # (preload < 0.5 and/or compliance < 0.7) so incidental chest-tube
    # placement for a hemothorax without tension doesn't over-shoot BP.
    if h.preload_index < 0.5:
        h.preload_index = lerp(h.preload_index, 0.85, 0.5)
    if h.compliance_index < 0.7:
        h.compliance_index = lerp(h.compliance_index, 0.9, 0.5)
    h.shunt_fraction = clip(h.shunt_fraction * 0.8, 0.0, 0.8)
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))


@intervention_rule("needle_decompress")
def _needle_decompress(h, flags, params):
    flags["needle_decompression"] = True
    # Tension pneumothorax relief: venous return restored, lung re-expands.
    # Only bump preload/compliance when they're clearly depressed (tension
    # physiology). Non-tension "prophylactic" needle placement (trauma
    # without true tension) is net-negative on BP MAE if we force high
    # floors here.
    if h.preload_index < 0.5:
        h.preload_index = lerp(h.preload_index, 0.85, 0.5)
    if h.compliance_index < 0.7:
        h.compliance_index = lerp(h.compliance_index, 0.85, 0.5)
    h.shunt_fraction = clip(h.shunt_fraction * 0.7, 0.0, 0.8)
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))


@intervention_rule("pericardiocentesis")
def _pericardiocentesis(h, flags, params):
    flags["pericardiocentesis"] = True
    # Pericardial drainage unloads the right heart: CO and preload recover.
    # Gate on depressed CO to avoid over-correction on low-severity pairs.
    if h.CO_index < 0.6:
        h.CO_index = lerp(h.CO_index, 0.8, 0.5)
    if h.preload_index < 0.6:
        h.preload_index = lerp(h.preload_index, 0.85, 0.5)


# ======================================================================
# Temperature (2)
# ======================================================================

@intervention_rule("start_warming")
def _start_warming(h, flags, params):
    flags["warming_active"] = True
    # Active warming: +0.05 °C/min dominates any cooling trend.
    h.core_temp_trend = max(h.core_temp_trend, 0.05)


@intervention_rule("start_cooling")
def _start_cooling(h, flags, params):
    flags["cooling_active"] = True
    # Active cooling: -0.03 °C/min dominates any warming.
    h.core_temp_trend = min(h.core_temp_trend, -0.03)


# ======================================================================
# Surgical airway & misc (3)
# ======================================================================

@intervention_rule("needle_cricothyroidotomy")
def _needle_cric(h, flags, params):
    # Temporizing cannula: opens an airway past complete obstruction.
    # Not a vent circuit, so FiO2 is device-dependent; the cannula itself
    # restores ability to oxygenate at whatever FiO2 the O2 source delivers.
    flags["airway"] = True
    # Bypassing airway obstruction: shunt improves substantially but
    # less than surgical cric (no PEEP/recruitment from PPV).
    h.shunt_fraction = clip(h.shunt_fraction * 0.5, 0.0, 0.8)
    h.compliance_index = max(h.compliance_index, 0.8)
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))


@intervention_rule("surgical_cricothyrotomy")
def _surgical_cric(h, flags, params):
    # Definitive surgical ETT: behaves exactly like intubate physiologically.
    flags["intubated"] = True
    flags["O2_device"] = "vent"
    flags["FiO2"] = 1.0
    if flags.get("vent_rate") is None:
        flags["vent_rate"] = 12
    if flags.get("vent_TV_ml") is None:
        flags["vent_TV_ml"] = 500
    h.shunt_fraction = clip(h.shunt_fraction * 0.6, 0.0, 0.8)
    h.compliance_index = max(h.compliance_index, 0.7)
    _oxygenate(h, 1.0, weight=0.55, duration_s=_duration(params))


@intervention_rule("escharotomy")
def _escharotomy(h, flags, params):
    # Circumferential burn chest eschar release: thoracic compliance restored.
    h.compliance_index = max(h.compliance_index, 0.9)
    h.shunt_fraction = clip(h.shunt_fraction * 0.8, 0.0, 0.8)
    fio2 = flags.get("FiO2") or 0.21
    _oxygenate(h, float(fio2), duration_s=_duration(params))
