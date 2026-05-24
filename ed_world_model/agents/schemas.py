"""Shared agent input and final-output schemas for v1.3.1 agents."""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


AgentRole = Literal["clinician", "nurse", "patient", "relative"]
ProfileTraitValue = str | int | float | bool | list[str] | None


class AgentProfile(BaseModel):
    """Role configuration passed to an agent before generation.

    Dynamic emotion, patient-internal truth, and conversation memory stay
    outside this model. Extra role style metadata belongs inside ``traits``.
    """

    model_config = ConfigDict(extra="forbid")

    role: AgentRole
    name: str | None = None
    traits: dict[str, ProfileTraitValue] = Field(default_factory=dict)


class AgentRuntimeInput(BaseModel):
    """Partial role-specific runtime context passed to an agent.

    ``observation`` is already filtered by ObservationBuilder. This model must
    not contain full GlobalState.
    """

    model_config = ConfigDict(extra="forbid")

    role: AgentRole
    observation: dict[str, Any]
    profile: AgentProfile | None = None
    emotion_context: dict[str, Any] | None = None
    recent_messages: list[dict[str, Any]] | None = None
    turn_index: NonNegativeInt | None = None


class VerbalAction(BaseModel):
    """Final spoken output from an agent.

    This is dialogue only. It does not represent internal reasoning and should
    only become a runtime message through StateManager.add_message(...).
    """

    model_config = ConfigDict(extra="forbid")

    speaker: str
    content: str
    recipient: str | None = None
    requires_response: bool = False

    @property
    def message_recipient(self) -> str | None:
        return self.recipient


class AgentProposal(BaseModel):
    """Final proposal returned by an agent.

    The proposal is final structured output only. It does not include internal
    reasoning or chain-of-thought.
    """

    model_config = ConfigDict(extra="forbid")

    verbal_action: VerbalAction | None = None
    action: dict[str, Any] | None = None


ClinicianProposal = AgentProposal


class VerbalOnlyProposal(BaseModel):
    """Final proposal shape for nurse, patient, and relative agents."""

    model_config = ConfigDict(extra="forbid")

    verbal_action: VerbalAction | None = None
    action: None = None


class Agent(Protocol):
    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        """Generate a final proposal or remain silent."""


__all__ = [
    "Agent",
    "AgentProfile",
    "AgentProposal",
    "AgentRole",
    "AgentRuntimeInput",
    "ClinicianProposal",
    "ProfileTraitValue",
    "VerbalAction",
    "VerbalOnlyProposal",
]
