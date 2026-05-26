"""Patient verbal-only prompt and parser boundary for v1.3.1."""
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


class PatientParserError(ValueError):
    """Raised when a patient LLM response is not valid final JSON output."""


class PatientAgent:
    """LLM-callable wrapper for the patient role.

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
        emotion_context: Mapping[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return build_patient_prompt(
            observation,
            profile=self._profile,
            emotion_context=emotion_context,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )

    def generate(
        self,
        observation: dict[str, Any],
        *,
        emotion_context: Mapping[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> AgentProposal:
        prompt = self.build_prompt(
            observation,
            emotion_context=emotion_context,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )
        output = self._llm_callable(prompt)
        return _to_agent_proposal(parse_patient_response(output))


def build_patient_prompt(
    observation: Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    emotion_context: Mapping[str, Any] | None = None,
    recent_messages: list[dict[str, Any]] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build a patient-only prompt from partial observation and profile context."""

    patient_profile = _coerce_profile(profile)
    messages = recent_messages
    if messages is None:
        messages = list(observation.get("recent_messages", []))

    sections = [
        "You are the patient in an ED simulation.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        "You only know what is in your partial observation.",
        (
            "Your observation may include patient_internal_state fields: "
            "chief_complaint, symptoms, hidden_history, hidden_allergies, "
            "hidden_home_medications, and disclosure_rules."
        ),
        (
            "hidden_* fields are patient-owned truth hidden from the clinical "
            "team, not hidden from you."
        ),
        (
            "Patient truthfulness: You are a source of patient-side truth. You "
            "must not invent symptoms, history, allergies, medications, social "
            "history, or review-of-systems findings."
        ),
        (
            "Only disclose information present in your observation, "
            "patient_internal_state, disclosure_rules, or recent conversation."
        ),
        (
            "Do not add plausible disease-associated symptoms just because "
            "they fit the diagnosis."
        ),
        (
            "If asked about a symptom or fact not present in your observation, "
            "including patient_internal_state, disclosure_rules, or recent "
            "conversation, answer conservatively: deny it if the observation "
            "says it is absent, say unsure or not mentioned if unknown, or "
            "stay silent if unable."
        ),
        (
            "hidden_* fields are patient-owned truth, but still must not be "
            "expanded beyond your observation."
        ),
        (
            "Do not expand hidden_history, hidden_allergies, or "
            "hidden_home_medications beyond what is shown."
        ),
        (
            "If you are unsure whether you know something, do not invent it."
        ),
        (
            "disclosure_rules are free-form guidance for what to volunteer and "
            "what to reveal only if asked."
        ),
        "Use the current emotion context if provided.",
        (
            "Follow communication ability flags; if you cannot speak, return "
            "null verbal_action or a minimal inability response only if allowed "
            "by the observation."
        ),
        (
            "Before deciding whether to speak, silently reason about whether a "
            "verbal contribution is useful, who it should target, and what "
            "content is appropriate based only on your observation, profile "
            "traits, emotion context if present, and recent messages."
        ),
        ANTI_REPETITION_RULE,
        (
            "Your verbal_action.content should reflect your AgentProfile traits "
            "when appropriate, such as communication style, anxiety, personality, "
            "health literacy, relationship context, or other provided traits."
        ),
        (
            "Profile traits shape how you speak; they must not cause you to "
            "invent clinical facts or reveal facts beyond your observation and "
            "disclosure guidance."
        ),
        VERBAL_REQUIRES_RESPONSE_RULE,
        (
            "Patient answers to clinician questions should normally use "
            "requires_response=false unless you explicitly ask a follow-up "
            "question."
        ),
        "You can answer questions, express symptoms, ask a short question, or stay silent.",
        "Do not provide medical orders.",
        "Do not invent facts outside your observation.",
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
            '    "speaker": "patient",\n'
            '    "target": "clinician" | "patient" | "nurse" | "relative" | null,\n'
            '    "content": "...",\n'
            '    "requires_response": true | false\n'
            "  } | null\n"
            "}"
        ),
    ]

    if turn_index is not None:
        sections.append(f"Turn index: {turn_index}")
    if patient_profile is not None:
        sections.append(f"Profile:\n{_format_json(patient_profile)}")
        if patient_profile.traits:
            sections.append(f"Profile traits:\n{_format_json(patient_profile.traits)}")
    if emotion_context is not None:
        sections.append(f"Patient emotion context:\n{_format_json(emotion_context)}")

    sections.extend(
        [
            f"Role-specific observation:\n{_format_json(observation)}",
            f"Recent messages:\n{_format_json(messages)}",
        ]
    )
    return "\n\n".join(sections)


def parse_patient_response(output: str | Mapping[str, Any]) -> VerbalOnlyProposal:
    """Parse final patient JSON into the shared verbal-only proposal model."""

    try:
        return parse_verbal_only_response(output, role="patient")
    except ValueError as exc:
        raise PatientParserError(f"Invalid patient response: {exc}") from exc


parse_patient_proposal = parse_patient_response


def _to_agent_proposal(proposal: VerbalOnlyProposal) -> AgentProposal:
    return AgentProposal(verbal_action=proposal.verbal_action, action=None)


def _coerce_profile(
    profile: AgentProfile | Mapping[str, Any] | None,
) -> AgentProfile | None:
    if profile is None:
        return None
    if isinstance(profile, AgentProfile):
        if profile.role != "patient":
            raise ValueError("Patient profile requires role='patient'.")
        return profile
    coerced = AgentProfile.model_validate(profile)
    if coerced.role != "patient":
        raise ValueError("Patient profile requires role='patient'.")
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
    "PatientAgent",
    "PatientParserError",
    "build_patient_prompt",
    "parse_patient_proposal",
    "parse_patient_response",
]
