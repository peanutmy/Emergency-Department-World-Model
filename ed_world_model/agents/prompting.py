"""Prompt construction and generic proposal parsing helpers for agents."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ed_world_model.agents.schemas import (
    AgentProposal,
    AgentRuntimeInput,
    AgentRole,
    VerbalOnlyProposal,
)


VERBAL_ONLY_ROLES = {"nurse", "patient", "relative"}
ALL_ROLES = {"clinician", *VERBAL_ONLY_ROLES}
FORBIDDEN_OUTPUT_KEYS = {"raw_text", "internal_reasoning", "chain_of_thought"}
VERBAL_ONLY_FORBIDDEN_OUTPUT_KEYS = {
    *FORBIDDEN_OUTPUT_KEYS,
    "action",
    "medical_treatment_order",
    "diagnostic_order",
    "behavior_action",
}
VERBAL_ONLY_RESPONSE_KEYS = {"verbal_action"}
VERBAL_ACTION_TARGETS = {"clinician", "patient", "nurse", "relative", None}
COMMON_PARTIAL_OBSERVATION_RULE = (
    "The following observation is a partial role-specific view of the world. "
    "Do not assume access to hidden state not shown here."
)
ANTI_REPETITION_RULE = (
    "Before speaking, silently compare your planned verbal_action.content with "
    "recent messages, especially your own prior messages. Avoid repeating the "
    "same information, reassurance, question, concern, or instruction in "
    "different words. If you have nothing new or useful to add, return "
    "verbal_action as null. Repetition is allowed when you are directly asked "
    "again, correcting a misunderstanding, confirming critical information, or "
    "new clinical or conversation context makes repetition necessary."
)
VERBAL_REQUIRES_RESPONSE_RULE = (
    "Set verbal_action.requires_response=true only for an explicit question "
    "that requires the target agent to answer. Use requires_response=false for "
    "ordinary answers, symptom statements, acknowledgments, status reports, "
    "reassurance, bedside instructions, and simple confirmations. If the "
    "content has no question mark and is not clearly asking for an answer, use "
    "requires_response=false."
)


def build_agent_prompt(runtime_input: AgentRuntimeInput | Mapping[str, Any]) -> str:
    """Build deterministic text instructions for a future LLM agent.

    This helper performs no model call. It only formats the already-filtered
    runtime input into role-specific instructions.
    """

    agent_input = _coerce_runtime_input(runtime_input)
    if agent_input.role == "clinician":
        return build_clinician_prompt(agent_input)
    if agent_input.role == "nurse":
        return build_nurse_prompt(agent_input)
    if agent_input.role == "patient":
        return build_patient_prompt(agent_input)
    if agent_input.role == "relative":
        return build_relative_prompt(agent_input)
    raise ValueError(f"Unsupported agent role: {agent_input.role}")


def build_clinician_prompt(
    runtime_input: AgentRuntimeInput | Mapping[str, Any],
) -> str:
    agent_input = _coerce_runtime_input(runtime_input, expected_role="clinician")
    return _compose_prompt(
        agent_input,
        role_instructions=[
            "Clinician output rules:",
            "- You may produce one verbal_action and one action.",
            "- action must be medical_treatment_order, diagnostic_order, or null.",
            "- no_action is system-generated and is not selectable by the clinician.",
            "- Do not include raw_text anywhere in the output.",
            "- A medical_treatment_order uses family -> kind_hint -> params.",
            "- A diagnostic_order selects one test_name only.",
            "- Action params may be null if unknown or unspecified.",
            "Clinician JSON shape:",
            (
                '{"verbal_action": {"speaker": "clinician", '
                '"recipient": "patient|nurse|relative|null", '
                '"content": "...", "requires_response": false}, '
                '"action": {"type": "medical_treatment_order", '
                '"family": "...", "kind_hint": "...", "params": {...}} '
                '| {"type": "diagnostic_order", "test_name": "..."} | null}'
            ),
        ],
    )


def build_nurse_prompt(runtime_input: AgentRuntimeInput | Mapping[str, Any]) -> str:
    agent_input = _coerce_runtime_input(runtime_input, expected_role="nurse")
    return _compose_prompt(
        agent_input,
        role_instructions=[
            "Nurse output rules:",
            "- The nurse is verbal-only in v1.3.1.",
            "- The nurse has no physical action and must not return an action.",
            "- The nurse can report result-related info or bedside reassurance.",
            f"- {VERBAL_REQUIRES_RESPONSE_RULE}",
            (
                "- Nurse reports to the clinician should normally use "
                "requires_response=false unless the nurse explicitly asks a "
                "question."
            ),
            "- Stay silent if there is no useful contribution.",
            "Nurse JSON shape:",
            (
                '{"verbal_action": {"speaker": "nurse", '
                '"recipient": "patient|clinician|relative|null", '
                '"content": "...", "requires_response": false}, '
                '"action": null}'
            ),
        ],
    )


def build_patient_prompt(runtime_input: AgentRuntimeInput | Mapping[str, Any]) -> str:
    agent_input = _coerce_runtime_input(runtime_input, expected_role="patient")
    return _compose_prompt(
        agent_input,
        emotion_label="Patient emotion context",
        role_instructions=[
            "Patient output rules:",
            (
                "- Use only the patient_internal_state subset present in the "
                "observation, plus visible messages and patient emotion."
            ),
            (
                "- disclosure_rules are free-form guidance, not deterministic "
                "rules."
            ),
            (
                "- Do not reveal hidden facts unless the observation and "
                "disclosure_rules allow it."
            ),
            (
                "- The patient can answer, be unable to answer, or stay silent "
                "depending on communication ability."
            ),
            f"- {VERBAL_REQUIRES_RESPONSE_RULE}",
            (
                "- Patient answers to clinician questions should normally use "
                "requires_response=false unless the patient explicitly asks a "
                "follow-up question."
            ),
            "Patient JSON shape:",
            (
                '{"verbal_action": {"speaker": "patient", '
                '"recipient": "clinician|nurse|relative|null", '
                '"content": "...", "requires_response": false}, '
                '"action": null}'
            ),
        ],
    )


def build_relative_prompt(
    runtime_input: AgentRuntimeInput | Mapping[str, Any],
) -> str:
    agent_input = _coerce_runtime_input(runtime_input, expected_role="relative")
    return _compose_prompt(
        agent_input,
        role_instructions=[
            "Relative output rules:",
            "- The relative is verbal-only in v1.3.1.",
            "- The relative has no behavior action.",
            f"- {VERBAL_REQUIRES_RESPONSE_RULE}",
            (
                "- Relative concerns should normally use "
                "requires_response=false unless the relative explicitly asks a "
                "question."
            ),
            "- Stay silent unless asked or explicitly selected.",
            "- There is no relative emotion transition in v1.3.1.",
            "Relative JSON shape:",
            (
                '{"verbal_action": {"speaker": "relative", '
                '"recipient": "clinician|nurse|patient|null", '
                '"content": "...", "requires_response": false}, '
                '"action": null}'
            ),
        ],
    )


def parse_agent_proposal(
    output: str | Mapping[str, Any],
    *,
    role: AgentRole,
) -> AgentProposal:
    """Parse final structured JSON into an AgentProposal.

    This is generic proposal parsing only. It intentionally does not validate
    clinician ``kind_hint`` values, diagnostic ``test_name`` values, or medical
    params; ActionValidator owns those checks later in the turn.
    """

    if role not in ALL_ROLES:
        raise ValueError(f"Unsupported agent role: {role}")

    data = _coerce_json_object(output)
    forbidden_key_path = _find_forbidden_key(data)
    if forbidden_key_path is not None:
        raise ValueError(
            "Agent proposals must not include raw_text, internal_reasoning, "
            f"or chain_of_thought fields: {forbidden_key_path}"
        )

    if role in VERBAL_ONLY_ROLES and data.get("action") is not None:
        raise ValueError(f"{role} proposals are verbal-only and cannot include action.")

    return AgentProposal.model_validate(data)


def parse_verbal_only_response(
    output: str | Mapping[str, Any],
    *,
    role: AgentRole,
) -> VerbalOnlyProposal:
    """Parse strict final JSON for nurse, patient, and relative agents.

    These agents may speak or stay silent. They must not return action fields or
    hidden reasoning fields. The LLM-facing schema uses ``target``; the shared
    runtime schema stores that value as ``recipient``.
    """

    if role not in VERBAL_ONLY_ROLES:
        raise ValueError(f"Role {role!r} is not a verbal-only agent role.")

    data = _coerce_strict_json_object(output)
    forbidden_key_path = _find_forbidden_verbal_only_key(data)
    if forbidden_key_path is not None:
        raise ValueError(
            "Verbal-only responses must not include action, "
            "medical_treatment_order, diagnostic_order, behavior_action, "
            "raw_text, internal_reasoning, or chain_of_thought fields: "
            f"{forbidden_key_path}"
        )

    if "verbal_action" not in data:
        raise ValueError("Verbal-only response must include verbal_action.")
    extra_keys = sorted(set(data) - VERBAL_ONLY_RESPONSE_KEYS)
    if extra_keys:
        raise ValueError(
            "Verbal-only response must contain only verbal_action; "
            f"found extra keys {extra_keys}."
        )

    normalized = {
        "verbal_action": _normalize_verbal_only_action(data["verbal_action"], role),
    }
    return VerbalOnlyProposal.model_validate(normalized)


def _compose_prompt(
    runtime_input: AgentRuntimeInput,
    *,
    role_instructions: list[str],
    emotion_label: str = "Emotion context",
) -> str:
    sections = [
        "You are generating the final proposal for an active ED simulation agent.",
        f"Role: {runtime_input.role}",
        COMMON_PARTIAL_OBSERVATION_RULE,
        (
            "The active agent may stay silent by returning null verbal_action "
            "and null action."
        ),
        ANTI_REPETITION_RULE,
        "Output final structured JSON only.",
        "Do not include internal reasoning in the output or store it.",
        "Do not include chain-of-thought in the output or store it.",
        "Do not append internal reasoning or chain-of-thought to runtime_state.messages.",
        "Do not include raw_text in the output.",
        "Only final verbal_action.content becomes conversation.",
    ]

    if runtime_input.turn_index is not None:
        sections.append(f"Turn index: {runtime_input.turn_index}")

    if runtime_input.profile is not None:
        sections.append(f"Profile:\n{_format_json(runtime_input.profile)}")
        if runtime_input.profile.traits:
            sections.append(
                f"Profile traits:\n{_format_json(runtime_input.profile.traits)}"
            )

    if runtime_input.emotion_context is not None:
        sections.append(
            f"{emotion_label}:\n{_format_json(runtime_input.emotion_context)}"
        )

    sections.append(
        f"Role-specific observation:\n{_format_json(runtime_input.observation)}"
    )
    sections.append(
        f"Recent messages:\n{_format_json(_recent_messages(runtime_input))}"
    )
    sections.extend(role_instructions)
    return "\n\n".join(sections)


def _recent_messages(runtime_input: AgentRuntimeInput) -> Any:
    if runtime_input.recent_messages is not None:
        return runtime_input.recent_messages
    return runtime_input.observation.get("recent_messages", [])


def _coerce_runtime_input(
    runtime_input: AgentRuntimeInput | Mapping[str, Any],
    *,
    expected_role: str | None = None,
) -> AgentRuntimeInput:
    if isinstance(runtime_input, AgentRuntimeInput):
        agent_input = runtime_input
    else:
        agent_input = AgentRuntimeInput.model_validate(runtime_input)
    if expected_role is not None and agent_input.role != expected_role:
        raise ValueError(
            f"{expected_role} prompt requires role={expected_role}, "
            f"got role={agent_input.role}."
        )
    return agent_input


def _coerce_json_object(output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("Agent proposal output must be valid JSON.") from exc
    elif isinstance(output, Mapping):
        data = dict(output)
    else:
        raise ValueError("Agent proposal output must be a JSON object.")

    if not isinstance(data, dict):
        raise ValueError("Agent proposal output must be a JSON object.")
    return data


def _coerce_strict_json_object(output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            data = json.loads(
                output,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Verbal-only response must be valid strict JSON.") from exc
    elif isinstance(output, Mapping):
        data = dict(output)
    else:
        raise ValueError("Verbal-only response must be a JSON object.")

    if not isinstance(data, dict):
        raise ValueError("Verbal-only response must be a JSON object.")
    return data


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value!r}.")


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_OUTPUT_KEYS:
                return current_path
            nested = _find_forbidden_key(item, current_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_key(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _find_forbidden_verbal_only_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = f"{path}.{key_text}"
            if key_text in VERBAL_ONLY_FORBIDDEN_OUTPUT_KEYS:
                return current_path
            nested = _find_forbidden_verbal_only_key(item, current_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_verbal_only_key(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _normalize_verbal_only_action(
    value: Any,
    role: AgentRole,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("verbal_action must be an object or null.")

    verbal_action = dict(value)
    has_target = "target" in verbal_action
    has_recipient = "recipient" in verbal_action
    if has_target and has_recipient:
        raise ValueError("verbal_action must not include both target and recipient.")
    if not has_target and not has_recipient:
        raise ValueError("verbal_action.target is required when verbal_action is not null.")
    if has_target:
        target = verbal_action.pop("target")
    else:
        target = verbal_action.pop("recipient")
    if target not in VERBAL_ACTION_TARGETS:
        raise ValueError(
            "verbal_action.target must be clinician, patient, nurse, relative, or null."
        )

    speaker = verbal_action.get("speaker")
    if speaker != role:
        raise ValueError(f"verbal_action.speaker must be {role!r}.")

    verbal_action["recipient"] = target
    requires_response = verbal_action.get("requires_response")
    if (
        requires_response not in (False, None)
        and not _is_explicit_targeted_question(
            verbal_action.get("content"),
            target=target,
        )
    ):
        verbal_action["requires_response"] = False
    return verbal_action


def _is_explicit_targeted_question(content: Any, *, target: Any) -> bool:
    if target is None or not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    first_word = stripped.lstrip("\"'([{").split(maxsplit=1)[0].lower()
    first_word = first_word.rstrip(":,.;!")
    return first_word in {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "would",
        "will",
        "do",
        "does",
        "did",
        "is",
        "are",
        "am",
        "was",
        "were",
        "have",
        "has",
        "had",
        "should",
        "may",
        "might",
    }


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
    "COMMON_PARTIAL_OBSERVATION_RULE",
    "ANTI_REPETITION_RULE",
    "VERBAL_REQUIRES_RESPONSE_RULE",
    "FORBIDDEN_OUTPUT_KEYS",
    "VERBAL_ONLY_ROLES",
    "VERBAL_ONLY_FORBIDDEN_OUTPUT_KEYS",
    "build_agent_prompt",
    "build_clinician_prompt",
    "build_nurse_prompt",
    "build_patient_prompt",
    "build_relative_prompt",
    "parse_agent_proposal",
    "parse_verbal_only_response",
]
