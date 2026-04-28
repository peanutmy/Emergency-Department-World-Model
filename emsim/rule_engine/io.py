"""
Schema-state ↔ HiddenState conversion (ENGINE_DESIGN.md §3.5).

`decode_state`: given a schema `state` dict, return (HiddenState, flags).
`encode_state`: given (HiddenState, flags, ...), return a schema `after` dict.

Phase 3b scope:
  - `decode_state` (from Phase 3a): two-step initialization (§3.5) —
    pathology-signature overlay on `DEFAULT_HEALTHY`, then residual
    correction of designated vars so `mapping.*_from_hidden` reproduces
    `before.vitals`.
  - `encode_state` (NEW in Phase 3b): routes **all vitals** through
    `mapping.*_from_hidden`. Residual correction in decode made round-trip
    near-exact for 96%+ of pairs, so for pure-wait pairs the output
    continues to match `before.vitals`. Action-modified pairs whose
    interventions update flags (e.g. `intubate → vent_rate=12`) now
    surface the flag-driven RR; pairs where `h.rhythm` was flipped
    (defibrillate VF → sinus) get the calibrated sinus HR baseline
    instead of pass-through 0.
    Arrest preservation:
      - asystole / VF → HR=0, BP=(0,0)
      - PEA           → HR=before_HR (monitor HR preserved), BP=(0,0)
      - O2Sat under arrest → mapping returns None, encode falls back to
        `before.O2Sat` (pulse-ox "last-valid-hold" per Phase 2 analysis)
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import mapping
from .hidden_state import ARREST_RHYTHMS, HiddenState
from .pathology_lib import (
    DEFAULT_HEALTHY,
    PATHOLOGY_INIT_SIGNATURES,
    severity_to_scalar,
)


# ======================================================================
# Rhythm inference (Phase 2 — unchanged)
# ======================================================================

def infer_rhythm(before: dict[str, Any]) -> str:
    """Heuristic rhythm inference from `before` state.

    Gate: `before.HR == 0` → arrest (subtype by pathology). Non-zero HR with
    pathology hints → VT / SVT / bradycardia. Default sinus.

    See Phase 2 report for why the `CPR_active` gate was removed (37 FP
    against 4 TP).
    """
    path = (before.get("mechanism", {}).get("pathology", {}) or {}).get("name", "").lower()
    hr = before.get("vitals", {}).get("HR", 0) or 0
    iv = before.get("interventions", {}) or {}
    cpr = bool(iv.get("CPR_active"))
    defib = iv.get("defib_last_J")

    if hr == 0:
        if "vf" in path or "ventricular_fibrillation" in path:
            return "VF"
        if "asystole" in path:
            return "asystole"
        if "pea" in path:
            return "PEA"
        if "arrest" in path or "vsa" in path:
            return "VF" if defib is not None else "PEA"
        return "PEA" if cpr else "asystole"

    # Note: pulseless-rhythm detection (BP=0 + HR>0 → PEA) was tried and
    # regressed eval: Phase 2 passed through monitor HR for these pairs,
    # and reclassifying as arrest zeroed them. Left as Phase 4 concern.

    if hr < 40:                                              # profound bradycardia
        return "bradycardia"
    if hr < 60 and ("brady" in path or "hyperkalemia" in path
                    or "beta_blocker" in path or "ccb" in path
                    or "digoxin" in path or "organophosphate" in path
                    or "hypothermia" in path):
        return "bradycardia"
    if hr > 150 and "svt" in path:
        return "SVT"
    if hr > 150 and ("ventricular_tachy" in path or path == "vt"
                     or "electrical_storm" in path):
        return "VT"

    return "sinus"


# ======================================================================
# Decode (Phase 3a — two-step initialization)
# ======================================================================

# Residual correction tuning knobs. These match the task's tolerance/2
# targets and don't need to be tighter — residual correction saturates
# well before these thresholds.
_RESIDUAL_CLIP = {
    "chronotropic_drive": (-4.0, 10.0),    # widened 7→10 so peds SVT HR=260
                                           # (drive=9 at +10·drive gain) fits
    "SVR_index":          (0.2,  4.0),
    "CO_index":           (0.05, 4.0),     # §3 says [0, 2.5] but residual
                                           # correction needs wider range to
                                           # fit extreme pulse pressures
                                           # (BP 240/120 → CO_eff=3.0).
    "ventilatory_drive":  (0.0,  5.0),     # peds RR can reach 60
    "PaO2_effective":     (0.0,  600.0),   # §3 says [20, 600] but allowing 0
                                           # so observed O2Sat=0 pairs (arrest)
                                           # inverse-map correctly.
}


def _apply_signature(h: HiddenState, sig: dict[str, Any]) -> None:
    for var, value in sig.items():
        if hasattr(h, var):
            setattr(h, var, value)


def _init_from_pathology(before: dict[str, Any]) -> HiddenState:
    """Step 1 (§3.5): start at DEFAULT_HEALTHY and overlay pathology signature."""
    h = HiddenState()
    _apply_signature(h, DEFAULT_HEALTHY)

    path = before.get("mechanism", {}).get("pathology", {}) or {}
    name = path.get("name", "")
    severity = path.get("severity", "moderate")
    sig_fn = PATHOLOGY_INIT_SIGNATURES.get(name)
    if sig_fn is not None:
        sig = sig_fn(severity_to_scalar(severity))
        _apply_signature(h, sig)
    # else: quietly fall back to DEFAULT_HEALTHY (warn path: see _unknown_pathologies)
    return h


# Track unknown pathologies once (avoid spam across 808 pairs).
_unknown_pathologies: set[str] = set()


def _residual_correct(h: HiddenState,
                      before: dict[str, Any],
                      flags: dict[str, Any]) -> None:
    """Step 2 (§3.5): fit designated hidden vars to match observed vitals.

    Designated pairings (§3.5 table, with a Phase 3a addition):
      HR     → chronotropic_drive      (skip if arrest)
      BP_sys → SVR_index               (paired with CO_index for dual-var solve)
      BP_dia → CO_index                (Phase 3a addition, deviates from §3.5
                                        which marked BP_dia "implicit from SVR";
                                        needed to hit round-trip tolerance/2 on
                                        pairs with narrow/wide pulse pressures)
      RR     → ventilatory_drive       (skip if intubated — vent_rate wins)
      O2Sat  → PaO2_effective          (inverse hemoglobin curve)
      T      — no residual (direct copy downstream)

    Rhythm is categorical and never residual-corrected (§3.5 explicit note).
    """
    bv = before["vitals"]

    # --- HR: chronotropic_drive (non-arrest only) ---
    if h.rhythm not in ARREST_RHYTHMS:
        target_hr = bv["HR"]
        if h.rhythm == "sinus":
            drive = (target_hr - mapping.HR_SINUS_BASELINE) / mapping.HR_SINUS_GAIN
        elif h.rhythm == "bradycardia":
            drive = (target_hr - mapping.HR_BRADY_BASELINE) / mapping.HR_BRADY_GAIN
        elif h.rhythm == "VT":
            drive = (target_hr - mapping.HR_VT_BASELINE) / mapping.HR_VT_GAIN
        elif h.rhythm == "SVT":
            drive = (target_hr - mapping.HR_SVT_BASELINE) / mapping.HR_SVT_GAIN
        else:
            drive = 0.0
        lo, hi = _RESIDUAL_CLIP["chronotropic_drive"]
        h.chronotropic_drive = mapping.clip(drive, lo, hi)

    # --- BP: analytic dual-var solve for (CO_eff, SVR) ---
    # Given BP_sys, BP_dia and the formula
    #     BP_sys = MAP_BASELINE * CO_eff * SVR + PP_HALF_BASELINE * CO_eff
    #     BP_dia = MAP_BASELINE * CO_eff * SVR - PP_HALF_BASELINE * CO_eff
    # Solve:
    #     CO_eff = (BP_sys - BP_dia) / (2 * PP_HALF_BASELINE)
    #     SVR    = (BP_sys + BP_dia) / (2 * MAP_BASELINE * CO_eff)
    # Then back out CO_index from CO_eff via frank_starling(preload_index).
    if h.rhythm not in ARREST_RHYTHMS:
        sys_v, dia_v = bv["BP_sys"], bv["BP_dia"]
        if sys_v > 0 and dia_v > 0 and sys_v > dia_v:
            pulse = sys_v - dia_v
            CO_eff = pulse / (2.0 * mapping.PP_HALF_BASELINE)
            SVR    = (sys_v + dia_v) / (2.0 * mapping.MAP_BASELINE * max(CO_eff, 1e-6))
            lo, hi = _RESIDUAL_CLIP["CO_index"]
            fs = mapping.frank_starling(h.preload_index) or 1.0
            h.CO_index = mapping.clip(CO_eff / max(fs, 1e-6), lo, hi)
            lo, hi = _RESIDUAL_CLIP["SVR_index"]
            h.SVR_index = mapping.clip(SVR, lo, hi)
        elif sys_v == 0 and dia_v == 0:
            # Pulseless non-arrest rhythm (pulseless VT/SVT): zero CO_eff
            # so `BP_from_hidden` returns (0, 0) without flipping rhythm to
            # arrest (encoding-side stays Phase-2-compatible).
            h.CO_index = 0.0
        # else: degenerate BP (sys<=dia or only one zero) — leave at signature.

    # --- RR: ventilatory_drive (when spontaneous breathing) ---
    # Gate mirrors `mapping.RR_from_hidden` in Phase 3b: only skip the fit
    # when vent_rate is authoritative (intubated AND vent_rate set). For
    # trach / BVM patients (intubated but vent_rate=None) we still fit
    # ventilatory_drive to before.RR so encode reproduces observed spontaneous RR.
    if not (flags.get("intubated") and flags.get("vent_rate") is not None):
        target_rr = bv["RR"]
        # Bypass consciousness gating during residual fit — rescue-breathing
        # and BVM scenarios have non-zero RR despite consciousness=0.
        drive = target_rr / mapping.RR_BASELINE_GAIN
        lo, hi = _RESIDUAL_CLIP["ventilatory_drive"]
        h.ventilatory_drive = mapping.clip(drive, lo, hi)

    # --- O2Sat: PaO2_effective via inverse curve ---
    target_o2 = bv["O2Sat"]
    lo, hi = _RESIDUAL_CLIP["PaO2_effective"]
    h.PaO2_effective = mapping.clip(mapping.inverse_hemoglobin_curve(target_o2), lo, hi)


def decode_state(before: dict[str, Any]) -> tuple[HiddenState, dict[str, Any]]:
    """Schema state → (HiddenState, flags).

    Two-step (§3.5):
      1. pathology signature overlay on DEFAULT_HEALTHY + rhythm override
         via `infer_rhythm` when arrest is detected
      2. residual correction of designated vars to fit observed vitals
    """
    # Step 1a: pathology signature
    h = _init_from_pathology(before)

    # Step 1b: rhythm reconciliation.
    #
    # `infer_rhythm` is **authoritative for arrest detection** because
    # `encode_state`'s arrest branch must match Phase 2 byte-for-byte.
    # If a signature set rhythm to an arrest type (e.g. pea_arrest → PEA)
    # but the data doesn't support it (before.HR > 0 → infer says sinus),
    # we must defer to infer_rhythm or encode will zero vitals that
    # Phase 2 passed through.
    #
    # For non-arrest, we keep the signature's rhythm if it's specifically
    # non-sinus (VT / SVT / bradycardia) — these carry info that encode
    # doesn't currently use but Phase 3b will. When signature was sinus
    # or an arrest type overridden to non-arrest, use infer_rhythm's
    # HR-based label.
    inferred = infer_rhythm(before)
    sig_rhythm = h.rhythm
    hr = before.get("vitals", {}).get("HR", 0) or 0
    if inferred in ARREST_RHYTHMS:
        h.rhythm = inferred
    elif sig_rhythm == "bradycardia" and hr < 60:
        h.rhythm = "bradycardia"
    elif sig_rhythm == "SVT" and hr > 140:
        h.rhythm = "SVT"
    elif sig_rhythm == "VT" and hr > 130:
        h.rhythm = "VT"
    else:
        h.rhythm = inferred

    # Track unregistered pathologies (single warning per name).
    path_name = (before.get("mechanism", {}).get("pathology", {}) or {}).get("name", "")
    if path_name and path_name not in PATHOLOGY_INIT_SIGNATURES \
            and path_name not in _unknown_pathologies:
        import warnings
        _unknown_pathologies.add(path_name)
        warnings.warn(
            f"pathology {path_name!r} has no registered init_signature — "
            f"falling back to DEFAULT_HEALTHY",
            RuntimeWarning,
            stacklevel=2,
        )

    # Step 1c: flags copy
    flags = deepcopy(before.get("interventions", {}))

    # Step 2: residual correction
    _residual_correct(h, before, flags)

    # Drug carry-over is no longer read from state — corpus has no drugs[]
    # field. Within-pair drug PD lives entirely on HiddenState.active_drug_effects,
    # populated by `drug_lib.start_drug` from pair.actions at t=0.
    return h, flags


# ======================================================================
# Encode (Phase 3a — UNCHANGED from Phase 2)
# ======================================================================
#
# Phase 3a discipline: even though `mapping.*_from_hidden` now returns real
# non-arrest values, we still pass through `before_dict` for non-arrest
# HR/BP/BP_dia and *always* for O2Sat/RR/T. Phase 3b will switch this.


def encode_state(h: HiddenState,
                 flags: dict[str, Any],
                 *,
                 T_before: float,
                 dt_s: float,
                 before_dict: dict[str, Any]) -> dict[str, Any]:
    """Schema `after` dict. Phase 3b: full vitals routing through mapping."""
    bv = before_dict["vitals"]

    # HR: arrest zeros via mapping (PEA preserved via before_HR argument).
    out_hr = mapping.HR_from_hidden(h, before_HR=bv["HR"])

    # BP: arrest → (0, 0); non-arrest from (CO_eff, SVR). Pulseless non-arrest
    # rhythms (VT/SVT with BP=0/0) have CO_index forced to 0 by the decoder,
    # so mapping returns (0, 0) for them without rhythm reclassification.
    out_sys, out_dia = mapping.BP_from_hidden(h, flags)

    # Safety net (BP narrow-PP): my BP formula `PP = 40·CO_eff` can't
    # represent `sys == dia > 0` (pulse=0 with non-zero MAP). 2 LVAD pairs
    # fall here (BP 50/50, 63/63) — residual didn't engage because the
    # analytic solve requires `sys > dia`. Pass through `before.BP` for
    # these pairs to avoid injecting the ~40-50 mmHg mis-fit error that
    # signature-default CO_eff/SVR would produce.
    before_pulse = bv["BP_sys"] - bv["BP_dia"]
    if bv["BP_sys"] > 0 and 0 <= before_pulse < 5:
        out_sys, out_dia = bv["BP_sys"], bv["BP_dia"]

    # RR: intubated → vent_rate from `flags`; else from ventilatory_drive.
    out_rr = mapping.RR_from_hidden(h, flags)

    # O2Sat: arrest returns None → pass through before.O2Sat (pulse-ox hold).
    out_o2 = mapping.O2Sat_from_hidden(h, flags)
    if out_o2 is None:
        out_o2 = bv["O2Sat"]

    # T: integrated trend (T_before + core_temp_trend * dt_min).
    # Phase 4: drift rules set `h.core_temp_trend` only for pathologies
    # with observed T movement (hypothermia passive warming, hyperthermia,
    # thyroid_storm, serotonin_syndrome). All other pathology drift rules
    # explicitly zero `h.core_temp_trend` so T is stable by default.
    # The Phase 3b safety-net filter is therefore removable.
    out_t = mapping.T_from_hidden(h, T_before, dt_s)

    out_vitals = {
        "HR":     out_hr,
        "BP_sys": out_sys,
        "BP_dia": out_dia,
        "RR":     out_rr,
        "O2Sat":  out_o2,
        "T":      out_t,
    }

    # Mechanism: copy pathology from before, expose current rhythm.
    # rhythm reflects post-action HiddenState (e.g., defibrillate VF→sinus,
    # start_pacing bradycardia→sinus, pathology drift sinus→PEA). The schema
    # accepts rhythm as optional, so existing tooling that doesn't read it
    # remains unaffected.
    out_mechanism = deepcopy(before_dict["mechanism"])
    out_mechanism["rhythm"] = h.rhythm

    return {
        "vitals":        out_vitals,
        "interventions": deepcopy(flags),
        "mechanism":     out_mechanism,
    }
