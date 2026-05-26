"""Relative verbal-only prompt and parser boundary for v1.3.1."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from ed_world_model.agents.prompting import (
    ANTI_REPETITION_RULE,
    COMMON_PARTIAL_OBSERVATION_RULE,
    FORBIDDEN_OUTPUT_KEYS,
    VERBAL_REQUIRES_RESPONSE_RULE,
    parse_verbal_only_response,
)
from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentProposal,
    VerbalOnlyProposal,
)


LLMCallable = Callable[[str], str]


class RelativeParserError(ValueError):
    """Raised when a relative LLM response is not valid final JSON output."""


class RelativeAgent:
    """LLM-callable wrapper for the relative role.

    The callable is injected so tests and runtime code can provide deterministic
    fakes. This class does not import, configure, or call any external LLM API.
    """

    def __init__(
        self,
        llm_callable: LLMCallable,
        *,
        profile: AgentProfile | Mapping[str, Any] | None = None,
    ) -> None:
        self._llm_callable = llm_callable
        self._profile = _coerce_profile(profile)

    def build_prompt(
        self,
        observation: Mapping[str, Any],
        *,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return build_relative_prompt(
            observation,
            profile=self._profile,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )

    def generate(
        self,
        observation: dict[str, Any],
        *,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> AgentProposal:
        prompt = self.build_prompt(
            observation,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )
        output = self._llm_callable(prompt)
        return _to_agent_proposal(parse_relative_response(output))


def build_relative_prompt(
    observation: Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    recent_messages: list[dict[str, Any]] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build a relative-only prompt from partial observation and profile context."""

    relative_profile = _coerce_profile(profile)
    messages = recent_messages
    if messages is None:
        messages = list(observation.get("recent_messages", []))

    sections = [
        "You are the patient's relative/family member in an ED simulation.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        "You are verbal-only in v1.3.1.",
        "You do not perform behavior actions or block care in v1.3.1.",
        "You may answer if asked, ask a brief question, express concern, or stay silent.",
        "You only know what is in your observation or what has been said in conversation.",
        "Do not invent hidden clinical facts.",
        "Do not read monitor-level exact vitals unless observation explicitly provides them.",
        "No relative emotion transition is modeled in v1.3.1.",
        (
            "Before deciding whether to speak, silently reason about whether a "
            "verbal contribution is useful, who it should target, and what "
            "content is appropriate based only on your observation, profile "
            "traits, and recent messages."
        ),
        ANTI_REPETITION_RULE,
        (
            "Your verbal_action.content should reflect your AgentProfile traits "
            "when appropriate, such as relationship role, concern level, health "
            "literacy, communication style, personality, or other provided traits."
        ),
        (
            "Profile traits shape how you speak; they must not cause you to "
            "invent hidden clinical facts, exact vitals, or care-blocking behavior."
        ),
        VERBAL_REQUIRES_RESPONSE_RULE,
        (
            "Relative concerns should normally use requires_response=false "
            "unless you explicitly ask a question."
        ),
        "Do not return action, medical_treatment_order, diagnostic_order, or behavior_action.",
        "Do not include raw_text.",
        "Do not include internal reasoning.",
        "Do not include chain-of-thought.",
        "Do not output the silent reasoning, rationale, or analysis.",
        "Output strict final structured JSON only.",
        (
            "Expected JSON response shape:\n"
            "{\n"
            '  "verbal_action": {\n'
            '    "speaker": "relative",\n'
            '    "target": "clinician" | "patient" | "nurse" | "relative" | null,\n'
            '    "content": "...",\n'
            '    "requires_response": true | false\n'
            "  } | null\n"
            "}"
        ),
    ]

    if turn_index is not None:
        sections.append(f"Turn index: {turn_index}")
    if relative_profile is not None:
        sections.append(f"Profile:\n{_format_json(relative_profile)}")
        if relative_profile.traits:
            sections.append(
                f"Profile traits:\n{_format_json(relative_profile.traits)}"
            )

    sections.extend(
        [
            f"Role-specific observation:\n{_format_json(observation)}",
            f"Recent messages:\n{_format_json(messages)}",
        ]
    )
    return "\n\n".join(sections)


def parse_relative_response(output: str | Mapping[str, Any]) -> VerbalOnlyProposal:
    """Parse final relative JSON into the shared verbal-only proposal model."""

    try:
        return parse_verbal_only_response(output, role="relative")
    except ValueError as exc:
        raise RelativeParserError(f"Invalid relative response: {exc}") from exc


parse_relative_proposal = parse_relative_response


def _to_agent_proposal(proposal: VerbalOnlyProposal) -> AgentProposal:
    return AgentProposal(verbal_action=proposal.verbal_action, action=None)


def _coerce_profile(
    profile: AgentProfile | Mapping[str, Any] | None,
) -> AgentProfile | None:
    if profile is None:
        return None
    if isinstance(profile, AgentProfile):
        if profile.role != "relative":
            raise ValueError("Relative profile requires role='relative'.")
        return profile
    coerced = AgentProfile.model_validate(profile)
    if coerced.role != "relative":
        raise ValueError("Relative profile requires role='relative'.")
    return coerced


def _format_json(value: Any) -> str:
    return json.dumps(_sanitize_for_prompt(value), indent=2, sort_keys=True)


def _sanitize_for_prompt(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _sanitize_for_prompt(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_for_prompt(item)
            for key, item in value.items()
            if str(key) not in FORBIDDEN_OUTPUT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_for_prompt(item) for item in value]
    return value


__all__ = [
    "LLMCallable",
    "RelativeAgent",
    "RelativeParserError",
    "build_relative_prompt",
    "parse_relative_proposal",
    "parse_relative_response",
]
