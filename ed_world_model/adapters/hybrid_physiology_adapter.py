"""Runtime adapter for the existing Hybrid physiology transition engine."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ed_world_model.state.global_state import GlobalState
from transition_engines.hybrid_engine import HybridEngine
from transition_engines.rule_based import RuleBasedEngine


class HybridPhysiologyAdapter:
    """Build v1.3.1 runtime engine input and call the Hybrid engine.

    The adapter reads only runtime ``GlobalState`` fields plus a validated
    engine-facing action. It does not load transition-pair data and does not
    mutate state.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        llm_client: Any | None = None,
    ) -> None:
        if engine is not None and llm_client is not None:
            raise ValueError("Provide either engine or llm_client, not both.")
        if engine is None:
            if llm_client is None:
                raise ValueError(
                    "HybridPhysiologyAdapter requires an engine or llm_client."
                )
            engine = HybridEngine(RuleBasedEngine(), llm_client)
        self._engine = engine

    def predict(
        self,
        global_state: GlobalState,
        engine_facing_action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return normalized vitals/features predicted by the Hybrid engine."""

        engine_input = self.build_engine_input(global_state, engine_facing_action)
        engine_output = self._engine.predict(engine_input)
        return self.normalize_engine_output(engine_output)

    @staticmethod
    def build_engine_input(
        global_state: GlobalState,
        engine_facing_action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build HybridEngine input from v1.3.1 runtime state only."""

        state = _model_dump(global_state)
        truth_state = _mapping(state.get("truth_state"))
        patient_state = _mapping(state.get("patient_state"))
        known_facts = _mapping(state.get("known_facts"))

        status_flags = deepcopy(_mapping(patient_state.get("status_flags")))

        return {
            "scenario_description": deepcopy(
                truth_state.get("scenario_description")
            ),
            "case_context": {
                "demographics": deepcopy(_mapping(truth_state.get("demographics"))),
                "patient_internal_state": deepcopy(
                    _mapping(truth_state.get("patient_internal_state"))
                ),
                "known_facts": deepcopy(known_facts),
                "patient_state": {"status_flags": status_flags},
            },
            "before": {
                "vitals": deepcopy(_mapping(patient_state.get("vitals"))),
                "features": deepcopy(_mapping(patient_state.get("features"))),
                "status_flags": deepcopy(status_flags),
            },
            "action": _runtime_action(engine_facing_action),
        }

    @staticmethod
    def normalize_engine_output(engine_output: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize HybridEngine output for later StateManager application."""

        if not isinstance(engine_output, Mapping):
            raise ValueError("Hybrid engine output must be an object.")

        prediction = _mapping(engine_output.get("prediction"))
        return {
            "vitals": deepcopy(_mapping(prediction.get("vitals"))),
            "features": deepcopy(_mapping(prediction.get("features"))),
            "metadata": deepcopy(_mapping(engine_output.get("metadata"))),
        }


def _runtime_action(engine_facing_action: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(engine_facing_action, Mapping):
        raise ValueError("engine_facing_action must be an object.")

    params = engine_facing_action.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ValueError("engine_facing_action.params must be an object if set.")

    return {
        "raw_text": None,
        "kind_hint": deepcopy(engine_facing_action.get("kind_hint")),
        "params": deepcopy(dict(params)),
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise ValueError("global_state must be a GlobalState-like object.")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["HybridPhysiologyAdapter"]
