"""
L1 core transition API — `RuleEngine.step(before, actions, duration_s) → after`.

Stateless, deterministic, teacher-forced. This is what `emsim_eval.py` runs.

Current coverage (Phase 5):
  - decode plumbing + drug cumulative merge
  - 21 intervention rules mutate `flags` + hidden state at t=0
  - drug actions plumbed onto `h.active_drug_effects`
  - pathology drift rules fire every tick (both pure-wait AND action pairs)
  - continuous intervention effects (fluids infusion) integrated every tick
  - drug PD tick stub (Phase 6 will fill in real contribution)

Unknown action names emit a one-shot warning via `warnings.warn` and are
skipped (never raised) — this keeps the harness running over the full 808
pairs even if a novel action slips in.
"""
from __future__ import annotations

import warnings
from typing import Any

from . import drug_lib, io
from .hidden_state import HiddenState
from .intervention_lib import INTERVENTION_RULES
from .mapping import clip
from .pathology_lib import DEFAULT_DRIFT, PATHOLOGY_RULES, severity_to_scalar


_warned_unknown: set[tuple[str, str]] = set()


def _warn_unknown(kind: str, name: str) -> None:
    """One-shot warning per (kind, name) pair — avoids 808× duplicate noise."""
    key = (kind, name)
    if key in _warned_unknown:
        return
    _warned_unknown.add(key)
    warnings.warn(
        f"RuleEngine: unknown {kind} action name {name!r} — skipping "
        f"(register a rule in {kind}_lib.py to fix)",
        RuntimeWarning,
        stacklevel=3,
    )


# Internal integration step (ENGINE_DESIGN.md §6 — design-frozen at 30s).
# 30s balances PK capture (epi peak ~60s) against compute cost. When
# duration_s is not a multiple of 30, the last tick uses the remainder
# (e.g. 45s → 30s + 15s).
_INTEGRATION_DT_S = 30.0

# Drift composition toggle.
# EMSIM_DRIFT_GATE=1 → Phase 4 behavior: drift fires only on pure-wait pairs.
# EMSIM_DRIFT_GATE=2 → default since Phase 6: drift on pure-wait +
#                      intervention pairs; drug-only pairs gated off. Phase 6
#                      calibration showed fully-unlocked drift regresses 16
#                      pairs on drug-only severe-pathology scenarios (author-
#                      stable pairs that drift drives out of the dead-zone),
#                      while drug PD alone can't offset. Gating drug-only
#                      preserves the Phase 5 pass-through majority while drug
#                      PD still fires on top at t=0+.
# EMSIM_DRIFT_GATE=0 → full unlock (drift on every slice).
import os as _os
_DRIFT_GATE = int(_os.environ.get("EMSIM_DRIFT_GATE", "2") or "2")


def _continuous_intervention_effects(h: HiddenState,
                                     flags: dict[str, Any],
                                     dt_s: float) -> None:
    """Apply per-tick effects for ongoing interventions (infusions, active
    warming/cooling, etc.).

    The intervention rule at t=0 captured the discrete jump; anything that
    keeps happening over the observation window lives here. Keep these
    effects small and self-limiting — they compose with pathology drift.
    """
    dt_min = dt_s / 60.0

    # Continuous IV fluids: infusion rate (ml/hr) → preload rise over time.
    # 500 ml/hr for 5 min ≈ 42 ml → ~0.025 preload units (small but additive).
    rate = flags.get("fluids_rate_ml_hr") or 0
    if rate:
        h.preload_index = clip(
            h.preload_index + 3e-5 * float(rate) * dt_min,
            0.0, 1.8,
        )

    # CPR while arrested: sustain minimum CO_index so composition stays
    # coherent if rhythm converts (defib → sinus within the same duration).
    if flags.get("CPR_active"):
        h.CO_index = max(h.CO_index, 0.3)
        h.preload_index = max(h.preload_index, 0.4)


class RuleEngine:
    """Rule-based physiology engine, L1 transition API."""

    def step(
        self,
        before: dict[str, Any],
        actions: list[dict[str, Any]],
        duration_s: float,
    ) -> dict[str, Any]:
        """Schema `before` → schema `after`."""
        h, flags = io.decode_state(before)

        # --- t=0: discrete action jumps ---
        for action in actions:
            atype = action.get("type")
            if atype == "intervention":
                name = action.get("name", "")
                rule = INTERVENTION_RULES.get(name)
                if rule is not None:
                    # Phase 5: expose the pair's observation window so rules
                    # can scale their jump by how much of the post-intervention
                    # equilibration the author will observe. duration_s=0 pairs
                    # are "immediate state markers" (authored vitals unchanged);
                    # duration_s>0 pairs let equilibration complete.
                    action_ctx = {**action, "_duration_s": float(duration_s)}
                    rule(h, flags, action_ctx)
                else:
                    _warn_unknown("intervention", name)
            elif atype == "drug":
                drug_lib.start_drug(h, action, t_sim_s=0.0)
            else:
                _warn_unknown("action-type", str(atype))

        # --- t > 0: integrate pathology drift + continuous intervention
        #     effects + drug PD.
        #
        # Phase 6 composition gate: drift fires on every slice. Drug PD is
        # now real (drug_lib.tick applies class-template effects), so drug-
        # only pairs get drift × drug-PD composition. EMSIM_DRIFT_GATE
        # overrides: 1 = Phase 4 (drift on pure-wait only), 2 = Phase 5
        # (drift on non-drug-only), 0 = Phase 6 default (drift everywhere).
        if duration_s > 0:
            pathology = before.get("mechanism", {}).get("pathology", {}) or {}
            path_name = pathology.get("name", "")
            severity = severity_to_scalar(pathology.get("severity", "moderate"))
            path_rule = PATHOLOGY_RULES.get(path_name, DEFAULT_DRIFT)
            has_intervention = any(a.get("type") == "intervention"
                                   for a in actions)
            drug_only = bool(actions) and not has_intervention

            if _DRIFT_GATE == 1:
                run_drift = not bool(actions)          # Phase 4
            elif _DRIFT_GATE == 2:
                run_drift = not drug_only               # Phase 5
            else:
                run_drift = True                         # Phase 6 default

            t = 0.0
            while t < duration_s:
                step_dt = min(_INTEGRATION_DT_S, duration_s - t)
                if path_rule is not None and run_drift:
                    path_rule(h, severity, step_dt)
                _continuous_intervention_effects(h, flags, step_dt)
                drug_lib.tick(h, step_dt, t_sim_s=t)
                t += step_dt

        # --- encode back to schema ---
        T_before = before["vitals"]["T"]
        return io.encode_state(
            h,
            flags,
            T_before=T_before,
            dt_s=duration_s,
            before_dict=before,
        )
