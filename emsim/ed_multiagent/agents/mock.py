"""Deterministic mock agent for tests and no-LLM orchestration."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..actions import AgentTurnOutput
from .base import BaseAgent


class MockAgent(BaseAgent):
    """Agent that returns a configured structured output without network calls."""

    role_instructions = (
        "Return the configured deterministic output.",
        "Do not infer facts beyond the provided observation.",
    )
    invariant_role_constraints = (
        "This mock is deterministic and makes no LLM call.",
        "It may be skipped by activation policy and must not assume per-round calls.",
    )

    def __init__(
        self,
        *,
        role: str = "mock",
        output: AgentTurnOutput | Mapping[str, Any] | None = None,
    ):
        super().__init__(role=role)
        self._output = self.validate_output(output or AgentTurnOutput())

    def act(self, observation: Any) -> AgentTurnOutput:
        self.build_static_prompt()
        self.build_dynamic_prompt(observation)
        return self.validate_output(deepcopy(self._output))


__all__ = ["MockAgent"]
