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
    VerbalDecision,
    VerbalOnlyProposal,
)
from ed_world_model.agents.verbal_decision import (
    VerbalDecisionParserError,
    parse_verbal_decision,
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
        llm_callable: LLMCallable | None = None,
        *,
        decision_llm_callable: LLMCallable | None = None,
        verbal_llm_callable: LLMCallable | None = None,
        profile: AgentProfile | Mapping[str, Any] | None = None,
    ) -> None:
        self._decision_llm_callable = decision_llm_callable or llm_callable
        self._verbal_llm_callable = verbal_llm_callable or llm_callable
        if self._decision_llm_callable is None or self._verbal_llm_callable is None:
            raise ValueError(
                "PatientAgent requires llm_callable or both decision and verbal callables."
            )
        self._profile = _coerce_profile(profile)
        self.last_verbal_decision: VerbalDecision | None = None

    def build_decision_prompt(
        self,
        observation: Mapping[str, Any],
        *,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return build_patient_decision_prompt(
            observation,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )

    def build_verbal_prompt(
        self,
        observation: Mapping[str, Any],
        decision: VerbalDecision | Mapping[str, Any],
        *,
        emotion_context: Mapping[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return build_patient_verbal_prompt(
            observation,
            decision,
            profile=self._profile,
            emotion_context=emotion_context,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )

    def build_prompt(
        self,
        observation: Mapping[str, Any],
        *,
        emotion_context: Mapping[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return self.build_decision_prompt(
            observation,
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
        self.last_verbal_decision = None
        decision_prompt = self.build_decision_prompt(
            observation,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )
        decision_output = self._decision_llm_callable(decision_prompt)
        decision = parse_patient_decision_response(decision_output)
        self.last_verbal_decision = decision
        if not decision.should_speak:
            return AgentProposal(verbal_action=None, action=None)

        verbal_prompt = self.build_verbal_prompt(
            observation,
            decision,
            emotion_context=emotion_context,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )
        verbal_output = self._verbal_llm_callable(verbal_prompt)
        return _to_agent_proposal(parse_patient_response(verbal_output))


def build_patient_decision_prompt(
    observation: Mapping[str, Any],
    *,
    recent_messages: list[dict[str, Any]] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build the patient Stage 1 VerbalDecision prompt."""

    decision_observation = _patient_decision_observation(
        observation,
        recent_messages=recent_messages,
    )

    sections = [
        "You are deciding whether and how the patient should speak.",
        "Stage 1: output a structured VerbalDecision JSON only.",
        "Do not output final dialogue here.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        (
            "Stage 1 does not use AgentProfile, traits, communication style, "
            "personality, or emotion_context. Do not use style or emotion to "
            "decide what facts should be said."
        ),
        (
            "Use only the patient observation, patient_internal_state, "
            "disclosure_rules, and recent conversation."
        ),
        (
            "You must not invent symptoms, history, allergies, medications, "
            "social history, or review-of-systems findings."
        ),
        (
            "If asked about unknown or unlisted facts, answer conservatively: "
            "deny it if the observation says it is absent, say unsure or not "
            "mentioned if unknown, or stay silent if unable."
        ),
        (
            "Use key_points for allowed facts or messages that may be included "
            "in final speech. Use forbidden_points for facts, symptoms, or "
            "claims the final speech must not add."
        ),
        "If no useful response is needed, set should_speak=false.",
        "Do not provide step-by-step reasoning.",
        "Do not include chain-of-thought or internal reasoning.",
        (
            "Provide only a short reasoning_summary grounded in the observation, "
            "ideally 1-2 sentences."
        ),
        (
            "Use recent messages to avoid repeating the same information, "
            "reassurance, question, concern, or instruction. If there is "
            "nothing new or useful to add, set should_speak=false."
        ),
        (
            "Expected JSON response shape:\n"
            "{\n"
            '  "speaker": "patient",\n'
            '  "should_speak": true | false,\n'
            '  "target": "clinician" | "nurse" | "patient" | "relative" | null,\n'
            '  "intent": "..." | null,\n'
            '  "reasoning_summary": "..." | null,\n'
            '  "key_points": ["..."],\n'
            '  "forbidden_points": ["..."],\n'
            '  "requires_response": true | false\n'
            "}"
        ),
    ]

    if turn_index is not None:
        sections.append(f"Turn index: {turn_index}")

    sections.append(
        f"Role-specific observation:\n{_format_json(decision_observation)}"
    )
    return "\n\n".join(sections)


def build_patient_verbal_prompt(
    observation: Mapping[str, Any],
    decision: VerbalDecision | Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    emotion_context: Mapping[str, Any] | None = None,
    recent_messages: list[dict[str, Any]] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build the patient Stage 2 final verbal-action prompt."""

    patient_profile = _coerce_profile(profile)
    verbal_decision = _coerce_patient_decision(decision)
    messages = recent_messages
    if messages is None:
        messages = list(observation.get("recent_messages", []))

    sections = [
        "You are the patient in an ED simulation.",
        "Stage 2: generate final verbal_action JSON only.",
        COMMON_PARTIAL_OBSERVATION_RULE,
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
            "patient_internal_state, disclosure_rules, recent conversation, "
            "or VerbalDecision key_points."
        ),
        (
            "disclosure_rules are free-form guidance for what to volunteer and "
            "what to reveal only if asked."
        ),
        (
            "hidden_* fields are patient-owned truth, but still must not be "
            "expanded beyond your observation."
        ),
        (
            "Do not expand hidden_history, hidden_allergies, or "
            "hidden_home_medications beyond what is shown."
        ),
        "If you are unsure whether you know something, do not invent it.",
        "Use the VerbalDecision as the source of truth for what to say.",
        (
            "You may use AgentProfile, traits, and emotion context only to "
            "shape wording and tone."
        ),
        (
            "Do not use profile, traits, or emotion to add new facts, symptoms, "
            "history, allergies, medications, social history, or test results."
        ),
        (
            "The final verbal_action.content may use only the VerbalDecision "
            "key_points and facts present in the patient observation."
        ),
        (
            "Do not include forbidden_points. Do not add plausible "
            "disease-associated symptoms just because they fit the diagnosis."
        ),
        (
            "If asked about a symptom or fact not present in your observation, "
            "including patient_internal_state, disclosure_rules, or recent "
            "conversation, answer conservatively: deny it if the observation "
            "says it is absent, say unsure or not mentioned if unknown, or "
            "stay silent if unable."
        ),
        ANTI_REPETITION_RULE,
        VERBAL_REQUIRES_RESPONSE_RULE,
        (
            "Patient answers to clinician questions should normally use "
            "requires_response=false unless you explicitly ask a follow-up "
            "question."
        ),
        "You can answer questions, express symptoms, ask a short question, or stay silent.",
        "Do not provide medical orders.",
        "Do not return action, medical_treatment_order, diagnostic_order, or behavior_action.",
        "Do not include raw_text.",
        "Do not include reasoning_summary.",
        "Do not include VerbalDecision.",
        "Do not include chain-of-thought or internal reasoning.",
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
            f"VerbalDecision:\n{_format_json(verbal_decision)}",
            f"Role-specific observation:\n{_format_json(observation)}",
            f"Recent messages:\n{_format_json(messages)}",
        ]
    )
    return "\n\n".join(sections)


def build_patient_prompt(
    observation: Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    emotion_context: Mapping[str, Any] | None = None,
    recent_messages: list[dict[str, Any]] | None = None,
    turn_index: int | None = None,
) -> str:
    """Compatibility wrapper for the patient Stage 2 prompt."""

    return build_patient_verbal_prompt(
        observation,
        VerbalDecision(
            speaker="patient",
            should_speak=True,
            target=None,
            intent="generate a patient verbal response if useful",
            reasoning_summary="Compatibility prompt without a separate decision.",
        ),
        profile=profile,
        emotion_context=emotion_context,
        recent_messages=recent_messages,
        turn_index=turn_index,
    )


def parse_patient_response(output: str | Mapping[str, Any]) -> VerbalOnlyProposal:
    """Parse final patient JSON into the shared verbal-only proposal model."""

    try:
        return parse_verbal_only_response(output, role="patient")
    except ValueError as exc:
        raise PatientParserError(f"Invalid patient response: {exc}") from exc


parse_patient_proposal = parse_patient_response


def parse_patient_decision_response(
    output: str | Mapping[str, Any],
) -> VerbalDecision:
    """Parse patient Stage 1 JSON into a transient VerbalDecision."""

    try:
        return parse_verbal_decision(output, speaker="patient")
    except VerbalDecisionParserError as exc:
        raise PatientParserError(f"Invalid patient verbal decision: {exc}") from exc


def _to_agent_proposal(proposal: VerbalOnlyProposal) -> AgentProposal:
    return AgentProposal(verbal_action=proposal.verbal_action, action=None)


def _coerce_patient_decision(
    decision: VerbalDecision | Mapping[str, Any],
) -> VerbalDecision:
    if isinstance(decision, VerbalDecision):
        if decision.speaker != "patient":
            raise ValueError("Patient VerbalDecision requires speaker='patient'.")
        return decision
    coerced = VerbalDecision.model_validate(decision)
    if coerced.speaker != "patient":
        raise ValueError("Patient VerbalDecision requires speaker='patient'.")
    return coerced


_PATIENT_DECISION_EXCLUDED_KEYS = {
    "agent_profile",
    "communication_style",
    "emotion_context",
    "patient_emotion",
    "profile",
    "traits",
}


def _patient_decision_observation(
    observation: Mapping[str, Any],
    *,
    recent_messages: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    sanitized = _drop_keys_for_decision(observation, _PATIENT_DECISION_EXCLUDED_KEYS)
    if recent_messages is not None:
        sanitized["recent_messages"] = _drop_keys_for_decision(
            recent_messages,
            _PATIENT_DECISION_EXCLUDED_KEYS,
        )
    return sanitized


def _drop_keys_for_decision(value: Any, excluded_keys: set[str]) -> Any:
    if hasattr(value, "model_dump"):
        return _drop_keys_for_decision(value.model_dump(), excluded_keys)
    if isinstance(value, Mapping):
        return {
            str(key): _drop_keys_for_decision(item, excluded_keys)
            for key, item in value.items()
            if str(key) not in excluded_keys
        }
    if isinstance(value, list):
        return [_drop_keys_for_decision(item, excluded_keys) for item in value]
    return value


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
    "build_patient_decision_prompt",
    "build_patient_prompt",
    "build_patient_verbal_prompt",
    "parse_patient_decision_response",
    "parse_patient_proposal",
    "parse_patient_response",
]
