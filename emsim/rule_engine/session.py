"""
L2 agent-facing API — `EngineSession` (ENGINE_DESIGN.md §6.5).

Phase 0: minimal stub (class exists, methods raise NotImplementedError).
Phase 0.5 will wire it as a thin wrapper around `RuleEngine.step`.
"""
from __future__ import annotations

from typing import Any


class EngineSession:
    """Stateful wrapper around `RuleEngine` — Phase 0.5 fills this in."""

    def __init__(self, initial_state: dict[str, Any]):
        self._initial_state = initial_state
        self._t_sim_s = 0.0
        self._event_queue: list = []

    def submit_action(self, agent_id: str, action: dict[str, Any]):
        raise NotImplementedError("Phase 0.5")

    def advance(self, duration_s: float):
        raise NotImplementedError("Phase 0.5")

    def observe(self, agent_id: str):
        raise NotImplementedError("Phase 0.5")

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError("Phase 0.5")
