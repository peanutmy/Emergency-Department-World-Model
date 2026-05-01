"""Base agent interface and prompt contract for M5."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from typing import Any

from ..actions import AgentTurnOutput
from ..observation import FORBIDDEN_OBSERVATION_KEYS


_DEFAULT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "AgentTurnOutput",
    "fields": {
        "verbal_action": "optional structured speech act",
        "information_action": "optional structured information report or request",
        "physical_action": (
            "optional structured intent only; deterministic workflow validates "
            "and executes physical tasks"
        ),
        "meta_action": "optional orchestration meta action when supported",
    },
}

_DEFAULT_SAFETY_RULES: tuple[str, ...] = (
    "Use only the role-specific observation supplied for the current turn.",
    "Do not invent vital signs, test results, diagnoses, or physiology.",
    "Do not access hidden diagnosis, hidden physiology, debug state, or engine internals.",
    "Do not mutate physiology or execute physical tasks directly.",
    "Physical execution remains deterministic through workflow and nurse-task systems.",
    "Clinical facts must remain structured; do not rely on prompt text as the only record.",
)

_DYNAMIC_WORKFLOW_KEYS = (
    "pending_orders",
    "active_task",
    "task_queue_summary",
    "queued_tasks",
    "completed_tasks",
    "pending_events",
)

_DYNAMIC_EMOTION_KEYS = (
    "emotion_state",
    "emotional_state",
    "emotional_or_social_context",
)


class BaseAgent:
    """Common interface for deterministic mock agents and future LLM agents."""

    role = "agent"
    role_instructions: tuple[str, ...] = ()
    invariant_role_constraints: tuple[str, ...] = ()
    allowed_action_schema: Mapping[str, Any] = _DEFAULT_ACTION_SCHEMA
    safety_rules: tuple[str, ...] = _DEFAULT_SAFETY_RULES

    def __init__(self, *, role: str | None = None):
        resolved_role = role if role is not None else self.role
        self.role = self._normalize_role(resolved_role)
        self._static_prompt_cache: str | None = None

    def build_static_prompt(self) -> str:
        """Build invariant role instructions, boundaries, and action schema."""

        if self._static_prompt_cache is None:
            self._static_prompt_cache = self._compose_static_prompt()
        return self._static_prompt_cache

    def build_dynamic_prompt(self, observation: Any) -> str:
        """Build the per-turn prompt from the current role observation only."""

        self._validate_observation_role(observation)
        observation_data = _sanitize_dynamic(_to_plain_data(observation))
        sections = [
            "Current role-specific observation:",
            _json_dumps(observation_data),
        ]

        recent_dialogue = _field_value(observation_data, "recent_dialogue")
        if recent_dialogue:
            sections.extend(("Recent dialogue:", _json_dumps(recent_dialogue)))

        response_opportunities = _field_value(
            observation_data,
            "response_opportunities",
        )
        if response_opportunities:
            sections.extend(
                ("Response opportunity:", _json_dumps(response_opportunities))
            )

        workflow_state = {
            key: _field_value(observation_data, key)
            for key in _DYNAMIC_WORKFLOW_KEYS
            if _has_dynamic_value(_field_value(observation_data, key))
        }
        if workflow_state:
            sections.extend(("Task / workflow state:", _json_dumps(workflow_state)))

        emotion_state = {
            key: _field_value(observation_data, key)
            for key in _DYNAMIC_EMOTION_KEYS
            if _has_dynamic_value(_field_value(observation_data, key))
        }
        if emotion_state:
            sections.extend(("Emotion state:", _json_dumps(emotion_state)))

        return "\n\n".join(sections)

    def act(self, observation: Any) -> AgentTurnOutput:
        """Return a deterministic no-op output for M5.

        Future LLM agents can reuse the same prompt methods before replacing
        this no-op implementation with a provider call.
        """

        self.build_static_prompt()
        self.build_dynamic_prompt(observation)
        return self.validate_output(self.default_output(observation))

    def validate_output(
        self,
        output: AgentTurnOutput | Mapping[str, Any],
    ) -> AgentTurnOutput:
        """Coerce and validate a structured agent turn output."""

        if isinstance(output, AgentTurnOutput):
            return output
        if isinstance(output, Mapping):
            return AgentTurnOutput.from_mapping(output)
        raise TypeError("agent output must be AgentTurnOutput or mapping")

    def default_output(self, observation: Any) -> AgentTurnOutput:
        _ = observation
        return AgentTurnOutput(metadata={"role": self.role, "no_llm": True})

    def _validate_observation_role(self, observation: Any) -> None:
        observed_role_value = _field_value(observation, "role")
        if observed_role_value is None:
            return

        observed_role = str(observed_role_value).strip().lower()
        if observed_role != self.role:
            raise ValueError(
                f"observation role {observed_role!r} does not match "
                f"agent role {self.role!r}"
            )

    def _compose_static_prompt(self) -> str:
        role_instructions = self.role_instructions or (
            "Follow the role-specific observation boundary.",
        )
        invariant_constraints = self.invariant_role_constraints or (
            "Do not assume the agent is called every simulation round.",
        )

        sections = [
            f"Role: {self.role}",
            "Role instructions:",
            _format_bullets(role_instructions),
            "Allowed action schema:",
            _json_dumps(_to_plain_data(self.allowed_action_schema)),
            "Safety / information-boundary rules:",
            _format_bullets(self.safety_rules),
            "Invariant role constraints:",
            _format_bullets(invariant_constraints),
        ]
        return "\n\n".join(sections)

    @staticmethod
    def _normalize_role(role: str) -> str:
        normalized = str(role).strip().lower()
        if not normalized:
            raise ValueError("agent role must be non-empty")
        return normalized


def _format_bullets(values: Sequence[str]) -> str:
    return "\n".join(f"- {str(value)}" for value in values)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _field_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _has_dynamic_value(value: Any) -> bool:
    return value not in (None, [], {}, (), set(), "")


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_data(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_plain_data(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain_data(item) for item in value]
    return deepcopy(value)


def _sanitize_dynamic(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_OBSERVATION_KEYS:
                continue
            sanitized[key_text] = _sanitize_dynamic(nested)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_dynamic(item) for item in value]
    return deepcopy(value)


__all__ = ["BaseAgent"]
