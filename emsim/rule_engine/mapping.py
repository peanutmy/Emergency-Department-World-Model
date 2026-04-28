"""
Hidden state → observable vitals mapping (ENGINE_DESIGN.md §4).

6 functions, one per vital. All coefficients are named constants so Phase 3
calibration can fit them against the 808 `before` states.

Phase 3a scope: full non-arrest formulas are implemented but `io.encode_state`
still passes through from `before_dict` for non-arrest outputs (Phase 3b will
flip the switch). The arrest branch is wired through encode_state as in
Phase 2 (HR=0, BP=(0,0); O2Sat always passes through — data preserves it).

Calibration target (Phase 3a): round-trip `decode_state(before) → mapping`
reproduces `before.vitals` within VITAL_TOLERANCE/2 on ≥95% of the 808 pairs.
Residual correction in `io.decode_state` handles per-pair fitting of the
designated vars (chronotropic_drive, SVR_index, CO_index, ventilatory_drive,
PaO2_effective); init_signatures set direction; the constants below set
baseline and gain.
"""
from __future__ import annotations

from typing import Any

from .hidden_state import ARREST_RHYTHMS, HiddenState


# ======================================================================
# Named coefficients (see ENGINE_DESIGN.md §4.1 — provenance comments)
# ======================================================================

# --- HR (plausible functional form; residual-corrected via chronotropic_drive) ---
# Baseline 80 bpm at drive=0 for a healthy adult; gain 20 bpm per SD-unit drive
# gives a ~[20, 180] sinus range at drive in [-3, +5] and avoids division by zero
# when residual correction clips.
HR_SINUS_BASELINE      = 80.0   # bpm at drive=0   [low-confidence form — calibrated Phase 3a]
HR_SINUS_GAIN          = 20.0   # bpm per unit drive
HR_SINUS_MIN, HR_SINUS_MAX        = 30.0, 230.0        # widened for peds + stress tachy
HR_VT_BASELINE, HR_VT_GAIN        = 180.0, 10.0
HR_VT_MIN, HR_VT_MAX              = 130.0, 260.0
HR_SVT_BASELINE, HR_SVT_GAIN      = 170.0, 10.0
HR_SVT_MIN, HR_SVT_MAX            = 140.0, 260.0
HR_BRADY_BASELINE, HR_BRADY_GAIN  = 45.0, 10.0
HR_BRADY_MIN, HR_BRADY_MAX        = 10.0, 80.0         # widened for pulseless idioventricular

# --- BP (real physiology: MAP = CO × SVR) ---
# At CO_eff=SVR=1.0, formula returns BP_sys=120, BP_dia=80 (textbook normal).
# Pulse-pressure term deliberately decoupled from SVR so dual-var residual
# correction (for BP_sys → SVR, BP_dia → CO_index) can reach an analytic fit.
MAP_BASELINE       = 100.0   # mmHg at CO=SVR=1.0   [low-confidence coefficient]
PP_HALF_BASELINE   = 20.0    # half pulse pressure at CO=1.0   [author-constructed proxy]

# --- Frank-Starling curve ---
# Returns preload → CO multiplier. Piecewise: rises to 1.0 at preload=1.0
# and plateaus around 1.15 by preload≈1.4. Clamped at 0 for negative preload.
def frank_starling(preload_index: float) -> float:
    p = max(0.0, preload_index)
    if p < 1.0:
        return p            # linear rise 0→1
    return min(1.15, 1.0 + 0.15 * (1.0 - (0.7 ** (p - 1.0))))


# --- RR ---
RR_BASELINE_GAIN = 14.0   # breaths/min at drive=consciousness=1.0
RR_MIN, RR_MAX   = 0.0, 50.0
VENT_RATE_DEFAULT = 12

# --- Hemoglobin dissociation curve (ENGINE_DESIGN.md §4 anchors) ---
# PaO2 (mmHg) → O2Sat (%). Piecewise-linear between anchors; clips [0, 100].
_HBO2_ANCHORS: list[tuple[float, float]] = [
    (0, 0), (20, 30), (40, 75), (60, 90), (80, 95),
    (100, 98), (150, 100), (600, 100),
]


def hemoglobin_curve(PaO2: float) -> float:
    """PaO2 (mmHg) → O2Sat (%); piecewise-linear, monotone non-decreasing."""
    if PaO2 <= _HBO2_ANCHORS[0][0]:
        return _HBO2_ANCHORS[0][1]
    if PaO2 >= _HBO2_ANCHORS[-1][0]:
        return _HBO2_ANCHORS[-1][1]
    for (x0, y0), (x1, y1) in zip(_HBO2_ANCHORS, _HBO2_ANCHORS[1:]):
        if x0 <= PaO2 <= x1:
            return y0 + (y1 - y0) * (PaO2 - x0) / (x1 - x0)
    return 100.0


def inverse_hemoglobin_curve(O2Sat: float) -> float:
    """O2Sat (%) → PaO2 (mmHg); piecewise-linear inverse of `hemoglobin_curve`.

    Used by `decode_state` residual correction to recover `PaO2_effective`
    from observed O2Sat. For saturations ≥100 returns 150 (clipping at
    upper plateau); for ≤0 returns 0.
    """
    sat = max(0.0, min(100.0, O2Sat))
    if sat <= _HBO2_ANCHORS[0][1]:
        return _HBO2_ANCHORS[0][0]
    if sat >= _HBO2_ANCHORS[-1][1]:
        # Multiple anchors collapse at 100%; return the lower one (150 mmHg)
        # so residual correction doesn't explode to 600.
        return 150.0
    for (x0, y0), (x1, y1) in zip(_HBO2_ANCHORS, _HBO2_ANCHORS[1:]):
        if y0 <= sat <= y1:
            if y1 == y0:
                return x0   # degenerate flat region: lowest PaO2 matching sat
            return x0 + (x1 - x0) * (sat - y0) / (y1 - y0)
    return 95.0


# ======================================================================
# Utility
# ======================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, weight: float) -> float:
    """Linear interpolation: weight=0 → a, weight=1 → b. Clamps weight to [0, 1]."""
    w = max(0.0, min(1.0, weight))
    return a + (b - a) * w


# ======================================================================
# Phase 2/3a public API — called from `io.encode_state` (arrest branch) AND
# from `io.decode_state` residual correction (non-arrest branch).
#
# Phase 3a signature: HR/BP/O2Sat return concrete values for non-arrest.
# `io.encode_state` still uses them as arrest-gate (treats non-arrest return
# as pass-through by construction) — Phase 3b will drop the gate.
# ======================================================================


def HR_from_hidden(h: HiddenState, before_HR: float = 0.0) -> float:
    """Hidden → HR (bpm).

    Arrest subtypes:
      - `asystole`, `VF`   → 0 (no organized depolarization, monitor shows 0)
      - `PEA`              → `before_HR` (§4 clarified semantic — PEA preserves
                              monitor HR; 38/43 dataset pairs confirm)

    Non-arrest: rhythm-specific linear function of `chronotropic_drive`,
    clipped to physiologic range.
    """
    if h.rhythm in ("asystole", "VF"):
        return 0.0
    if h.rhythm == "PEA":
        return float(before_HR) if before_HR is not None else 0.0
    if h.rhythm == "VT":
        return clip(HR_VT_BASELINE + HR_VT_GAIN * h.chronotropic_drive,
                    HR_VT_MIN, HR_VT_MAX)
    if h.rhythm == "SVT":
        return clip(HR_SVT_BASELINE + HR_SVT_GAIN * h.chronotropic_drive,
                    HR_SVT_MIN, HR_SVT_MAX)
    if h.rhythm == "bradycardia":
        return clip(HR_BRADY_BASELINE + HR_BRADY_GAIN * h.chronotropic_drive,
                    HR_BRADY_MIN, HR_BRADY_MAX)
    return clip(HR_SINUS_BASELINE + HR_SINUS_GAIN * h.chronotropic_drive,
                HR_SINUS_MIN, HR_SINUS_MAX)


def BP_from_hidden(h: HiddenState, flags: dict[str, Any] | None = None) -> tuple[float, float]:
    """Hidden → (BP_sys, BP_dia) (mmHg). Arrest → (0, 0)."""
    if h.rhythm in ARREST_RHYTHMS:
        return (0.0, 0.0)
    CO_eff = h.CO_index * frank_starling(h.preload_index)
    MAP    = MAP_BASELINE * CO_eff * h.SVR_index
    pulse  = PP_HALF_BASELINE * CO_eff * 2       # = PP_BASELINE * CO_eff
    return (MAP + pulse / 2, MAP - pulse / 2)


def O2Sat_from_hidden(h: HiddenState, flags: dict[str, Any] | None = None) -> float | None:
    """Hidden → O2Sat (%).

    Arrest rhythms return **None** so `io.encode_state` passes through
    `before.O2Sat`. Phase 2 report: 21/50 `before.HR==0` pairs carry
    preserved `before.O2Sat > 0` (pulse-ox "last-valid-hold") into
    `after.O2Sat > 0`; zeroing would regress MAE. Keeping pass-through
    semantics under arrest costs 1 correct-zero pair but saves 21.

    Non-arrest: hemoglobin dissociation curve of `PaO2_effective`.
    """
    if h.rhythm in ARREST_RHYTHMS:
        return None
    return hemoglobin_curve(h.PaO2_effective)


def RR_from_hidden(h: HiddenState, flags: dict[str, Any]) -> float:
    """Hidden → RR (breaths/min).

    Ventilator override: `intubated=True AND vent_rate is not None` →
    return `vent_rate` (mechanical ventilation dominates). When `intubated=True`
    but `vent_rate is None` (tracheostomy with NRB, BVM without set rate),
    fall through to spontaneous `ventilatory_drive` — `intubated` alone is
    not sufficient to override the patient's own rate. 5 dataset pairs
    depend on this distinction (pediatric_drowning p7/p8, tracheostomy_
    emergency p1/p2/p3).

    Non-intubated or intubated-w/o-vent-rate: spontaneous RR from
    `ventilatory_drive`. Consciousness multiplier was removed in Phase 3a
    (see module docstring).
    """
    if flags.get("intubated") and flags.get("vent_rate") is not None:
        return float(flags["vent_rate"])
    return clip(RR_BASELINE_GAIN * h.ventilatory_drive, RR_MIN, RR_MAX)


def T_from_hidden(h: HiddenState, T_before: float, dt_s: float) -> float:
    return T_before + h.core_temp_trend * (dt_s / 60.0)


def vitals_from_hidden(h: HiddenState, flags: dict[str, Any],
                       T_before: float, dt_s: float) -> dict[str, float]:
    """Full 6-vital dict. Phase 3b+ `io.encode_state` may call this directly."""
    bp_sys, bp_dia = BP_from_hidden(h, flags)
    return {
        "HR":     HR_from_hidden(h, flags),
        "BP_sys": bp_sys,
        "BP_dia": bp_dia,
        "RR":     RR_from_hidden(h, flags),
        "O2Sat":  O2Sat_from_hidden(h, flags),
        "T":      T_from_hidden(h, T_before, dt_s),
    }
