"""
L3 orchestrator protocol (ENGINE_DESIGN.md §6.5 — interface only in v1).

No implementation: defining the shape lets future multi-agent code import
`OrchestratorProtocol` without a circular dependency on a real orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .events import Event
from .session import EngineSession


@dataclass
class AgentTurn:
    agent_id: str
    action: dict[str, Any] | None   # None ⇒ "just advance time"
    duration_s: float = 0.0


class OrchestratorProtocol(Protocol):
    def next_turn(self, session: EngineSession) -> AgentTurn: ...
    def dispatch_events(self, events: list[Event]) -> None: ...
    def filter_observation(self, agent_id: str, full_state: dict[str, Any]) -> dict[str, Any]: ...
