"""Stateful EMSim physiology session.

This adapter is intentionally narrow: it keeps EMSim physiology alive across
simulated time and exposes vitals through an agent-facing observation method.
It does not implement workflow, memory, agents, LLM calls, or orchestration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import warnings

try:  # Support running tests from either the repository root or emsim/.
    from rule_engine import drug_lib, io
    from rule_engine.engine import (
        _INTEGRATION_DT_S,
        _continuous_intervention_effects,
    )
    from rule_engine.hidden_state import ARREST_RHYTHMS
    from rule_engine.intervention_lib import INTERVENTION_RULES
    from rule_engine.pathology_lib import (
        DEFAULT_DRIFT,
        PATHOLOGY_RULES,
        severity_to_scalar,
    )
except ImportError:  # pragma: no cover - exercised when imported as emsim.*
    from emsim.rule_engine import drug_lib, io
    from emsim.rule_engine.engine import (
        _INTEGRATION_DT_S,
        _continuous_intervention_effects,
    )
    from emsim.rule_engine.hidden_state import ARREST_RHYTHMS
    from emsim.rule_engine.intervention_lib import INTERVENTION_RULES
    from emsim.rule_engine.pathology_lib import (
        DEFAULT_DRIFT,
        PATHOLOGY_RULES,
        severity_to_scalar,
    )


@dataclass
class PhysiologyEvent:
    """Physiology-level signal emitted by ``advance``.

    M1a keeps this contract small so later orchestration can subscribe to it
    without requiring workflow or agent code now.
    """

    kind: str
    time_s: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineAdvanceResult:
    """Return value for ``EngineSession.advance``."""

    state: dict[str, Any]
    events: list[PhysiologyEvent] = field(default_factory=list)


class EngineSession:
    """Live wrapper around EMSim's deterministic physiology rules.

    The session stores the EMSim hidden state internally and only exposes
    vitals through agent-safe methods. Agents should use ``observe`` or
    ``observe_vitals``; neither returns hidden physiology or full schema state.
    """

    def __init__(self, initial_state: dict[str, Any], scenario_seed: int | None = None):
        self._initial_state = deepcopy(initial_state)
        self._state = deepcopy(initial_state)
        self._hidden_state, self._intervention_flags = io.decode_state(self._state)
        self._now_s = 0.0
        self._scenario_seed = scenario_seed
        self._pending_events: list[PhysiologyEvent] = []

    @property
    def now_s(self) -> float:
        """Current simulated time in seconds."""

        return self._now_s

    @property
    def current_state(self) -> dict[str, Any]:
        """Current full/debug EMSim schema state, returned as a defensive copy.

        This may include mechanism and intervention fields and must not be used
        as agent input. Use ``observe`` or ``observe_vitals`` for agent-safe
        views.
        """

        return deepcopy(self._state)

    def observe_vitals(self) -> dict[str, Any]:
        """Return an agent-safe vitals-only surface."""

        return deepcopy(self._state["vitals"])

    def observe(self, agent_id: str | None = None) -> dict[str, Any]:
        """Return the M1a agent-safe observation.

        ``agent_id`` is reserved for M1b role-specific views and ignored in
        M1a; every caller receives the same vitals-only view.
        """

        _ = agent_id
        return {
            "time_s": self._now_s,
            "vitals": self.observe_vitals(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a full/debug snapshot for deterministic tests/tools.

        The nested state may include mechanism and intervention fields and must
        not be used as agent input.
        """

        return {
            "time_s": self._now_s,
            "state": self.current_state,
        }

    def submit_action(self, agent_id: str, action: dict[str, Any]) -> None:
        """Compatibility alias for callers using an agent/action shape.

        ``agent_id`` is reserved for M1b role-aware routing and ignored in M1a.
        """

        _ = agent_id
        self.apply_emsim_action(action)

    def apply_emsim_action(self, action: dict[str, Any]) -> None:
        """Apply one EMSim-compatible action at the current simulated time."""

        action_copy = deepcopy(action)
        action_type = action_copy.get("type")
        before_rhythm = self._hidden_state.rhythm
        if action_type == "intervention":
            self._apply_intervention(action_copy)
        elif action_type == "drug":
            drug_lib.start_drug(
                self._hidden_state,
                action_copy,
                t_sim_s=self._now_s,
            )
        else:
            self._warn_unknown("action-type", str(action_type))
            return

        self._encode_current_state(dt_s=0.0)
        self._pending_events.extend(
            self._rhythm_events(
                before_rhythm=before_rhythm,
                after_rhythm=self._hidden_state.rhythm,
                time_s=self._now_s,
            )
        )

    def advance(self, dt_s: float) -> EngineAdvanceResult:
        """Advance live EMSim physiology by ``dt_s`` simulated seconds."""

        dt_s = float(dt_s)
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        if dt_s == 0:
            return EngineAdvanceResult(state=self.current_state, events=[])

        before_rhythm = self._hidden_state.rhythm

        self._integrate(dt_s)
        self._now_s += dt_s
        self._encode_current_state(dt_s=dt_s)

        events = [
            *self._pending_events,
            *self._rhythm_events(
                before_rhythm=before_rhythm,
                after_rhythm=self._hidden_state.rhythm,
                time_s=self._now_s,
            ),
        ]
        self._pending_events.clear()
        return EngineAdvanceResult(state=self.current_state, events=events)

    def _rhythm_events(
        self,
        *,
        before_rhythm: str,
        after_rhythm: str,
        time_s: float,
    ) -> list[PhysiologyEvent]:
        events: list[PhysiologyEvent] = []

        if before_rhythm != after_rhythm:
            events.append(
                PhysiologyEvent(
                    kind="rhythm_change",
                    time_s=time_s,
                    payload={"from": before_rhythm, "to": after_rhythm},
                )
            )

        if before_rhythm not in ARREST_RHYTHMS and after_rhythm in ARREST_RHYTHMS:
            events.append(
                PhysiologyEvent(
                    kind="arrest",
                    time_s=time_s,
                    payload={"rhythm": after_rhythm},
                )
            )

        if before_rhythm in ARREST_RHYTHMS and after_rhythm not in ARREST_RHYTHMS:
            events.append(
                PhysiologyEvent(
                    kind="rosc",
                    time_s=time_s,
                    payload={"from": before_rhythm, "to": after_rhythm},
                )
            )

        return events

    def _apply_intervention(self, action: dict[str, Any]) -> None:
        name = action.get("name", "")
        rule = INTERVENTION_RULES.get(name)
        if rule is None:
            self._warn_unknown("intervention", name)
            return

        action_context = {**action, "_duration_s": 0.0}
        rule(self._hidden_state, self._intervention_flags, action_context)

    def _integrate(self, duration_s: float) -> None:
        pathology = self._state.get("mechanism", {}).get("pathology", {}) or {}
        path_name = pathology.get("name", "")
        severity = severity_to_scalar(pathology.get("severity", "moderate"))
        path_rule = PATHOLOGY_RULES.get(path_name, DEFAULT_DRIFT)

        elapsed_s = 0.0
        while elapsed_s < duration_s:
            step_dt = min(_INTEGRATION_DT_S, duration_s - elapsed_s)
            if path_rule is not None:
                path_rule(self._hidden_state, severity, step_dt)
            _continuous_intervention_effects(
                self._hidden_state,
                self._intervention_flags,
                step_dt,
            )
            drug_lib.tick(
                self._hidden_state,
                step_dt,
                t_sim_s=self._now_s + elapsed_s,
            )
            elapsed_s += step_dt

    def _encode_current_state(self, dt_s: float) -> None:
        self._state = io.encode_state(
            self._hidden_state,
            self._intervention_flags,
            T_before=self._state["vitals"]["T"],
            dt_s=dt_s,
            before_dict=self._state,
        )

    @staticmethod
    def _warn_unknown(kind: str, name: str) -> None:
        warnings.warn(
            f"EngineSession: unknown {kind} action name {name!r} - skipping",
            RuntimeWarning,
            stacklevel=3,
        )
