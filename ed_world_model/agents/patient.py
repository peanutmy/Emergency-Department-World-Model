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
    build_json_repair_prompt,
    json_failure_snippet,
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

_PATIENT_DECISION_REPAIR_SHAPE = """{
  "speaker": "patient",
  "should_speak": true | false,
  "target": "clinician" | "nurse" | "patient" | "relative" | null,
  "intent": "..." | null,
  "reasoning_summary": "..." | null,
  "key_points": ["..."],
  "forbidden_points": ["..."],
  "requires_response": true | false
}"""
_PATIENT_VERBAL_REPAIR_SHAPE = """{
  "verbal_action": {
    "speaker": "patient",
    "target": "clinician" | "patient" | "nurse" | "relative" | null,
    "content": "...",
    "requires_response": true | false
  } | null
}"""


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
        recent_messages: list[dict[str, Any]] | None = None,
        turn_index: int | None = None,
    ) -> str:
        return build_patient_verbal_prompt(
            observation,
            decision,
            profile=self._profile,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )

    def build_prompt(
        self,
        observation: Mapping[str, Any],
        *,
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
        decision = self._parse_decision_with_repair(
            decision_prompt,
            decision_output,
        )
        self.last_verbal_decision = decision
        if not decision.should_speak:
            return AgentProposal(verbal_action=None, action=None)

        verbal_prompt = self.build_verbal_prompt(
            observation,
            decision,
            recent_messages=recent_messages,
            turn_index=turn_index,
        )
        verbal_output = self._verbal_llm_callable(verbal_prompt)
        return _to_agent_proposal(
            self._parse_verbal_with_repair(verbal_prompt, verbal_output)
        )

    def _parse_decision_with_repair(
        self,
        prompt: str,
        output: str | Mapping[str, Any],
    ) -> VerbalDecision:
        try:
            return parse_patient_decision_response(output)
        except PatientParserError as first_error:
            repair_prompt = build_json_repair_prompt(
                stage="patient Stage 1 VerbalDecision",
                original_prompt=prompt,
                invalid_output=output,
                error_message=str(first_error),
                expected_shape=_PATIENT_DECISION_REPAIR_SHAPE,
            )
            repair_output = self._decision_llm_callable(repair_prompt)
            try:
                return parse_patient_decision_response(repair_output)
            except PatientParserError as repair_error:
                raise PatientParserError(
                    f"{first_error}; repair retry failed: {repair_error}; "
                    f"raw={json_failure_snippet(output)}; "
                    f"repair_raw={json_failure_snippet(repair_output)}"
                ) from first_error

    def _parse_verbal_with_repair(
        self,
        prompt: str,
        output: str | Mapping[str, Any],
    ) -> VerbalOnlyProposal:
        try:
            return parse_patient_response(output)
        except PatientParserError as first_error:
            repair_prompt = build_json_repair_prompt(
                stage="patient Stage 2 final verbal_action",
                original_prompt=prompt,
                invalid_output=output,
                error_message=str(first_error),
                expected_shape=_PATIENT_VERBAL_REPAIR_SHAPE,
            )
            repair_output = self._verbal_llm_callable(repair_prompt)
            try:
                return parse_patient_response(repair_output)
            except PatientParserError as repair_error:
                raise PatientParserError(
                    f"{first_error}; repair retry failed: {repair_error}; "
                    f"raw={json_failure_snippet(output)}; "
                    f"repair_raw={json_failure_snippet(repair_output)}"
                ) from first_error


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
            "personality, or patient_emotion. Do not use style or emotion to "
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
            "Direct patient engagement: If the most recent visible message is "
            "addressed to the patient and asks a question, and communication_ability "
            "shows the patient is alive, conscious, and can_speak=True, default "
            "to should_speak=true. Answer with only facts available in the "
            "patient observation."
        ),
        (
            "If the most recent visible message is addressed to the patient "
            "and gives a treatment explanation, instruction, reassurance, or "
            "bedside action while the patient is still symptomatic or unstable, "
            "you may set should_speak=true for a brief acknowledgment, symptom "
            "update, or concern. Keep it short and grounded."
        ),
        (
            "If recent clinician/nurse messages are not addressed to the patient "
            "and do not require the patient's answer, usually set should_speak=false."
        ),
        (
            "If communication_ability shows the patient is not alive, not "
            "conscious, or can_speak=False, set should_speak=false."
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
            "You may use AgentProfile, traits, and the patient_emotion in your "
            "observation only to shape wording and tone."
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
