"""Turn-start active-agent selection for the ED world model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ed_world_model.state.global_state import GlobalState


CLINICIAN = "clinician"
NURSE = "nurse"
PATIENT = "patient"
RELATIVE = "relative"

DEFAULT_CLINICIAN_ACTIVE = "default_clinician_active"
REQUIRED_RESPONSE = "required_response"
NEWLY_AVAILABLE_RESULTS = "newly_available_results"
EXPLICITLY_SELECTED = "explicitly_selected"

AGENT_ORDER = (CLINICIAN, NURSE, PATIENT, RELATIVE)
EXPLICITLY_SELECTABLE_AGENTS = (RELATIVE,)


@dataclass(frozen=True)
class OrchestratorDecision:
    active_agents: list[str]
    activation_reasons: dict[str, list[str]]
    required_response_agents: list[str]


class Orchestrator:
    """Selects which agents get a turn-start opportunity."""

    def select_active_agents(
        self,
        global_state: GlobalState,
        explicitly_selected_agents: set[str] | None = None,
    ) -> OrchestratorDecision:
        runtime_state = global_state.runtime_state
        required_response_agents = _unique_in_agent_order(
            runtime_state.required_response_agents
        )
        explicit_agents = _unique_in_agent_order(explicitly_selected_agents or set())
        activation_reasons: dict[str, list[str]] = {}

        _add_reason(activation_reasons, CLINICIAN, DEFAULT_CLINICIAN_ACTIVE)

        for agent in required_response_agents:
            _add_reason(activation_reasons, agent, REQUIRED_RESPONSE)

        if runtime_state.newly_available_results:
            _add_reason(activation_reasons, NURSE, NEWLY_AVAILABLE_RESULTS)

        for agent in explicit_agents:
            if agent in EXPLICITLY_SELECTABLE_AGENTS:
                _add_reason(activation_reasons, agent, EXPLICITLY_SELECTED)

        active_agents = [
            agent for agent in AGENT_ORDER if agent in activation_reasons
        ]
        return OrchestratorDecision(
            active_agents=active_agents,
            activation_reasons=activation_reasons,
            required_response_agents=required_response_agents,
        )


def select_active_agents(
    global_state: GlobalState,
    explicitly_selected_agents: set[str] | None = None,
) -> OrchestratorDecision:
    return Orchestrator().select_active_agents(
        global_state,
        explicitly_selected_agents=explicitly_selected_agents,
    )


def _unique_in_agent_order(agent_names: Iterable[str]) -> list[str]:
    requested = set(agent_names)
    return [agent for agent in AGENT_ORDER if agent in requested]


def _add_reason(
    activation_reasons: dict[str, list[str]],
    agent: str,
    reason: str,
) -> None:
    reasons = activation_reasons.setdefault(agent, [])
    if reason not in reasons:
        reasons.append(reason)


__all__ = [
    "AGENT_ORDER",
    "CLINICIAN",
    "DEFAULT_CLINICIAN_ACTIVE",
    "EXPLICITLY_SELECTED",
    "EXPLICITLY_SELECTABLE_AGENTS",
    "NEWLY_AVAILABLE_RESULTS",
    "NURSE",
    "Orchestrator",
    "OrchestratorDecision",
    "PATIENT",
    "RELATIVE",
    "REQUIRED_RESPONSE",
    "select_active_agents",
]
