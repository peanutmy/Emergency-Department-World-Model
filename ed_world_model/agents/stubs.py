"""Deterministic stub agents for the M6 turn loop."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ed_world_model.agents.schemas import AgentProposal, VerbalAction


ScriptedProposal = AgentProposal | VerbalAction | dict[str, Any] | str | None


class SilentAgent:
    """Agent that always stays silent and proposes no action."""

    def __init__(self) -> None:
        self.generated_observations: list[dict[str, Any]] = []

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(observation)
        return AgentProposal()


class ScriptedClinicianAgent:
    """Return clinician proposals in script order, then silence."""

    def __init__(self, script: Iterable[ScriptedProposal]) -> None:
        self._script = list(script)
        self.generated_observations: list[dict[str, Any]] = []
        self._index = 0

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(observation)
        return _next_scripted_proposal(self, speaker="clinician")


class ScriptedPatientAgent:
    """Return scripted patient verbal responses in order."""

    def __init__(self, script: Iterable[ScriptedProposal]) -> None:
        self._script = list(script)
        self.generated_observations: list[dict[str, Any]] = []
        self._index = 0

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(observation)
        proposal = _next_scripted_proposal(self, speaker="patient")
        return AgentProposal(verbal_action=proposal.verbal_action)


class ScriptedNurseAgent:
    """Return scripted nurse verbal responses in order; never physical actions."""

    def __init__(self, script: Iterable[ScriptedProposal]) -> None:
        self._script = list(script)
        self.generated_observations: list[dict[str, Any]] = []
        self._index = 0

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(observation)
        proposal = _next_scripted_proposal(self, speaker="nurse")
        return AgentProposal(verbal_action=proposal.verbal_action)


class ScriptedRelativeAgent:
    """Return scripted relative verbal responses in order; never behavior actions."""

    def __init__(self, script: Iterable[ScriptedProposal]) -> None:
        self._script = list(script)
        self.generated_observations: list[dict[str, Any]] = []
        self._index = 0

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(observation)
        proposal = _next_scripted_proposal(self, speaker="relative")
        return AgentProposal(verbal_action=proposal.verbal_action)


def _next_scripted_proposal(agent: Any, *, speaker: str) -> AgentProposal:
    if agent._index >= len(agent._script):
        return AgentProposal()
    item = agent._script[agent._index]
    agent._index += 1
    return _coerce_scripted_proposal(item, speaker=speaker)


def _coerce_scripted_proposal(
    item: ScriptedProposal,
    *,
    speaker: str,
) -> AgentProposal:
    if item is None:
        return AgentProposal()
    if isinstance(item, AgentProposal):
        return item
    if isinstance(item, VerbalAction):
        return AgentProposal(verbal_action=item)
    if isinstance(item, str):
        return AgentProposal(
            verbal_action=VerbalAction(speaker=speaker, content=item)
        )
    if isinstance(item, dict):
        if "verbal_action" in item or "action" in item:
            return AgentProposal.model_validate(item)
        if "type" in item:
            return AgentProposal(action=item)
        if "speaker" in item and "content" in item:
            return AgentProposal(verbal_action=VerbalAction.model_validate(item))
    return AgentProposal.model_validate(item)


__all__ = [
    "ScriptedClinicianAgent",
    "ScriptedNurseAgent",
    "ScriptedPatientAgent",
    "ScriptedRelativeAgent",
    "SilentAgent",
]
