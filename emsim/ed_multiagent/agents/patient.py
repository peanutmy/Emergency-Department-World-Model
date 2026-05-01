"""Patient agent skeleton for M5."""
from __future__ import annotations

from typing import Any

from ..actions import AgentTurnOutput
from .base import BaseAgent


class PatientAgent(BaseAgent):
    """No-LLM patient skeleton with speech-capacity gating."""

    role = "patient"
    role_instructions = (
        "You are the patient.",
        "You know symptoms, feelings, and private memory only.",
        "You do not know exact vitals unless staff told you.",
        "You do not know hidden diagnosis.",
        "If speech_capacity is unable, you cannot answer verbally.",
    )
    invariant_role_constraints = (
        "Do not expose hidden physiology or undiscovered clinical facts.",
        "Do not invent vital signs or test results.",
        "Answer only from subjective state, recent dialogue, response opportunities, and private memory.",
        "Do not assume you are called every simulation round.",
    )

    def act(self, observation: Any) -> AgentTurnOutput:
        self.build_static_prompt()
        self.build_dynamic_prompt(observation)
        if _speech_capacity_value(observation) == "unable":
            return AgentTurnOutput(
                verbal_action=None,
                meta_action={
                    "type": "unable_to_answer",
                    "response_mode": "unable",
                    "reason": "speech_capacity_unable",
                },
                metadata={"role": self.role, "no_llm": True},
            )
        return self.validate_output(self.default_output(observation))


def _speech_capacity_value(observation: Any) -> str | None:
    subjective_state = _field_value(observation, "subjective_state")
    if subjective_state is None:
        return None
    capacity = _field_value(subjective_state, "speech_capacity")
    if capacity is None:
        return None
    return str(capacity).strip().lower()


def _field_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


__all__ = ["PatientAgent"]
