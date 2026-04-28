"""
Hidden-state dataclasses for the rule-based physiology engine.

The 12-variable hidden state (ENGINE_DESIGN.md §3) is the engine's internal
representation of the patient's physiology. Observable vitals are a pure
function of this state (see mapping.py); rules only ever mutate the hidden
state, never the vitals directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- Rhythm vocabulary (string enum) ---
RHYTHMS = ("sinus", "SVT", "VT", "VF", "asystole", "PEA", "bradycardia")
ARREST_RHYTHMS = frozenset({"asystole", "VF", "PEA"})


@dataclass
class DrugEffect:
    """A single time-resolved pharmacodynamic contribution (ENGINE_DESIGN.md §3).

    PD curve (two-phase): linear rise from 0 → ``magnitude`` over ``onset_s``,
    then exponential decay with half-life ``half_life_s``. Past ~5 half-lives
    the contribution is numerically zero; the entry is kept in the list but
    stops contributing.

    ``t_start_s`` is the simulation-time origin for the clock; the current
    age is ``t_sim_s - t_start_s``. Drugs scheduled by ``start_drug`` at t=0
    use ``t_start_s = 0``.

    ``_prev_contrib`` is the last contribution already applied to the target
    hidden var. ``drug_lib.tick`` applies only the *delta* between the new
    contribution and ``_prev_contrib`` each tick, then updates the field —
    this keeps hidden state coherent across the integration loop.
    """
    target_var: str                  # name of HiddenState field, e.g. "SVR_index"
    magnitude: float                 # signed peak contribution at t = t_start_s + onset_s
    t_start_s: float                 # simulation-time origin for the PK clock
    half_life_s: float               # exponential half-life of the effect
    onset_s: float = 60.0            # linear-rise time from 0 → magnitude
    drug_name: str = ""              # originating drug (for debugging / overrides)
    _prev_contrib: float = 0.0       # last contribution applied (for delta-apply)

    def contribution_at(self, age_s: float) -> float:
        """Signed contribution at PK clock age ``age_s`` (seconds since dose)."""
        if age_s <= 0 or self.magnitude == 0.0:
            return 0.0
        if age_s < self.onset_s:
            # Linear ramp from 0 to magnitude over onset_s.
            return self.magnitude * (age_s / self.onset_s)
        if self.half_life_s <= 0:
            return self.magnitude
        # Exponential decay from magnitude after onset completes.
        elapsed = age_s - self.onset_s
        if elapsed > 5.0 * self.half_life_s:
            return 0.0
        return self.magnitude * (0.5 ** (elapsed / self.half_life_s))


@dataclass
class HiddenState:
    """The 12 hidden physiology variables. Ranges documented in ENGINE_DESIGN.md §3."""
    rhythm: str = "sinus"                       # enum string
    CO_index: float = 1.0                       # [0, 2.5]
    SVR_index: float = 1.0                      # [0.3, 2.5]
    preload_index: float = 1.0                  # [0, 1.8]
    chronotropic_drive: float = 0.0             # [-2, +3] SD units
    PaO2_effective: float = 95.0                # mmHg, [20, 600]
    shunt_fraction: float = 0.05                # [0, 0.8]
    compliance_index: float = 1.0               # [0.2, 1.2]
    ventilatory_drive: float = 1.0              # [0, 2.5]
    core_temp_trend: float = 0.0                # °C/min
    consciousness: float = 1.0                  # [0, 1]
    active_drug_effects: list[DrugEffect] = field(default_factory=list)
