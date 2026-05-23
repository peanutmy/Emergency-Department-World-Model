"""Lightweight agent proposal schemas for stubbed v1.3.1 agents."""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


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
    """Final proposal returned by a stub/scripted agent."""

    model_config = ConfigDict(extra="forbid")

    verbal_action: VerbalAction | None = None
    action: dict[str, Any] | None = None


class Agent(Protocol):
    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        """Generate a final proposal or remain silent."""


__all__ = ["Agent", "AgentProposal", "VerbalAction"]
