"""Clinician LLM prompt and parser boundary for v1.3.1."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ed_world_model.actions.registry import ActionRegistry
from ed_world_model.agents.prompting import (
    ANTI_REPETITION_RULE,
    COMMON_PARTIAL_OBSERVATION_RULE,
    FORBIDDEN_OUTPUT_KEYS,
    parse_agent_proposal,
)
from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentRuntimeInput,
    ClinicianProposal,
)


LLMCallable = Callable[[str], str]
ACTION_TYPES = {"medical_treatment_order", "diagnostic_order"}
TOP_LEVEL_ACTION_ALIASES = {"actions", "medical_treatment_order", "diagnostic_order"}
TOP_LEVEL_KEYS = {"verbal_action", "action"}
DIAGNOSTIC_ORDER_KEYS = {"type", "test_name"}
MEDICAL_TREATMENT_ORDER_KEYS = {"type", "family", "kind_hint", "params"}


class ClinicianParserError(ValueError):
    """Raised when a clinician LLM response is not valid final JSON output."""


class ClinicianAgent:
    """LLM-callable wrapper for the clinician role.

    The callable is injected so tests and runtime code can provide deterministic
    fakes. This class does not import, configure, or call any external LLM API.
    """

    def __init__(
        self,
        llm_callable: LLMCallable,
        *,
        profile: AgentProfile | Mapping[str, Any] | None = None,
        action_registry: ActionRegistry | None = None,
    ) -> None:
        self._llm_callable = llm_callable
        self._profile = _coerce_profile(profile)
        self._action_registry = action_registry or ActionRegistry()

    def build_prompt(self, observation: Mapping[str, Any]) -> str:
        return build_clinician_prompt(
            observation,
            profile=self._profile,
            action_registry=self._action_registry,
        )

    def generate(self, observation: dict[str, Any]) -> ClinicianProposal:
        prompt = self.build_prompt(observation)
        output = self._llm_callable(prompt)
        return parse_clinician_response(output)


def build_clinician_prompt(
    observation: Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    action_registry: ActionRegistry | None = None,
    available_diagnostic_test_names: Sequence[str] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build a clinician-only prompt from partial observation and menu context."""

    registry = action_registry or ActionRegistry()
    runtime_input = AgentRuntimeInput(
        role="clinician",
        observation=dict(observation),
        profile=_coerce_profile(profile),
        turn_index=turn_index,
    )
    diagnostic_test_names = list(
        available_diagnostic_test_names
        if available_diagnostic_test_names is not None
        else _extract_available_diagnostic_test_names(observation)
    )

    sections = [
        "You are the clinician in an ED simulation.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        "You receive only a partial role-specific observation.",
        "Do not assume hidden state not shown.",
        "You may output at most one structured action per turn.",
        "You may produce at most one verbal_action and at most one action.",
        "You may stay silent if no useful verbal contribution.",
        (
            "If you produce a verbal_action, first silently reason about what "
            "the clinician would say and who they would address based only on "
            "the observation, recent messages, and clinician profile/traits."
        ),
        (
            "Use clinician AgentProfile traits to shape only communication style: "
            "tone, concision, confidence, bedside manner, explanation depth, "
            "teaching style, and who you address."
        ),
        (
            "Profile traits must not change observed facts, available diagnostic "
            "results, allowed actions, or action validation. Clinical decisions "
            "must remain grounded in the role-specific observation, released "
            "results, available action menu, and available diagnostic test names."
        ),
        (
            "If you produce a clinician action, first silently reason about "
            "which single medical_treatment_order or diagnostic_order fits "
            "based only on the observation, released results, available action "
            "menu, and available diagnostic test names."
        ),
        (
            "Diagnostic result grounding: When referencing diagnostic results, "
            "use only the exact test names and results present in the current "
            "observation. If a result is newly available or already available, "
            "refer to it by its exact displayed test name. Do not say that one "
            "diagnostic result is available when the available result belongs "
            "to a different test. Do not infer or substitute a different test "
            "name."
        ),
        (
            "If a diagnostic test is pending, describe it as pending/in "
            "progress rather than available. If a diagnostic result is not "
            "present in the observation, do not claim it is ready or available."
        ),
        (
            "Diagnostic result use: Before ordering a diagnostic test, check "
            "pending diagnostic results and available diagnostic results in "
            "the observation, including pending_diagnostic_results if shown, "
            "known_facts.available_results, and newly_available_results. Do "
            "not request or order a diagnostic test that is already pending."
        ),
        (
            "If a diagnostic test is pending, refer to it as pending or in "
            "progress rather than ordering it again. Do not request or order a "
            "diagnostic test whose result is already available. If a result is "
            "already available, use the available result instead of ordering "
            "the same test again. If a diagnostic result is available, use it "
            "in your reasoning or communication instead of ordering the same "
            "test again."
        ),
        (
            "Your verbal_action must not claim a duplicate test is being "
            "ordered if the structured action cannot validly order it."
        ),
        (
            "Numeric vital fidelity and rationale grounding: When referencing "
            "vital signs, use the values exactly as shown in the observation "
            "and use the exact values shown in the observation. Do not describe "
            "a value as low, high, hypotensive, hypertensive, tachycardic, "
            "bradycardic, hypoxic, or worsening unless the displayed current "
            "value or visible trend supports that description. If unsure, "
            "state the numeric value without interpreting it. If uncertain, "
            "state the numeric value without interpreting it. When explaining "
            "an action, make the verbal reason consistent with the observed "
            "vitals, known facts, pending results, and available results. Your "
            "verbal explanation for an action must be consistent with the "
            "observed vitals and known facts. Do not invent a rationale that "
            "is not supported by the observation."
        ),
        (
            "Emergency stabilization: If the patient has immediately dangerous "
            "airway/breathing/circulation findings, you may order urgent "
            "stabilization first, such as oxygen support, ventilation support, "
            "airway management, monitoring, or urgent diagnostics. Do not "
            "delay life-saving care for a long history checklist."
        ),
        (
            "Focused history: If the patient is conscious and can_speak=True, "
            "ask one brief focused question when clinically appropriate, "
            "especially early in the encounter. Prefer asking the patient "
            "directly for symptoms/onset. Ask nurse/chart/relative for "
            "objective course, medications, allergies, PMH, or if the patient "
            "cannot answer."
        ),
        (
            "A focused history question is not an additional clinical action. "
            "It may be included as the single verbal_action even when the "
            "structured action is an urgent treatment or diagnostic order, as "
            "long as it does not promise another unstructured test or treatment."
        ),
        (
            "Early patient contact: On the first 1-2 turns, if the patient is "
            "conscious and can_speak=True, and chief complaint, symptoms/onset, "
            "allergies, PMH, or home medications are missing from known_facts, "
            "strongly prefer a patient-directed verbal_action with "
            "requires_response=true."
        ),
        (
            "If urgent ABC stabilization is needed, do not delay the structured "
            "action. Pair the urgent action with one brief patient question "
            "when possible."
        ),
        (
            "Deterioration response: If the patient remains unstable or is "
            "worsening despite prior interventions, prioritize reassessment "
            "and stabilizing or escalating actions over non-urgent history "
            "questions. Do not continue asking non-urgent history questions "
            "while vital signs are worsening unless the information is "
            "immediately necessary for the next action."
        ),
        (
            "Information-dependent treatment: Before diagnosis-specific or "
            "medication-heavy treatments, consider whether enough key facts "
            "are known: chief complaint, symptoms/onset/progression, "
            "allergies, PMH, home medications, contraindications, and relevant "
            "results. If key facts are missing and the patient is stable "
            "enough, ask one focused question instead of guessing."
        ),
        (
            "One-question limit: Ask at most one focused history question per "
            "turn. Do not ask a long checklist in one message."
        ),
        (
            "Do not output this reasoning, rationale, analysis, "
            "internal_reasoning, or chain-of-thought."
        ),
        ANTI_REPETITION_RULE,
        (
            "action must be one of: medical_treatment_order, diagnostic_order, "
            "or null."
        ),
        (
            "verbal_action must be consistent with the single structured action "
            "for this turn."
        ),
        (
            "verbal_action must not promise, order, prepare, or describe "
            "additional tests or treatments that are not represented by the "
            "structured action for this turn."
        ),
        (
            "If action.type is medical_treatment_order, verbal_action may "
            "explain that treatment, provide general reassurance, or ask one "
            "focused patient history question relevant to the current problem."
        ),
        (
            "If action.type is diagnostic_order, verbal_action may explain only "
            "that one diagnostic test, provide general reassurance, or ask one "
            "focused patient history question relevant to the current problem."
        ),
        (
            "If action is null, verbal_action must not claim that a test or "
            "treatment is being started, ordered, prepared, or performed."
        ),
        (
            'Do not say things like "start oxygen and prepare ECG and chest '
            'X-ray" when only one structured action is allowed.'
        ),
        (
            "If more than one clinical step is needed, choose the single most "
            "important action for this turn and leave the rest for future turns."
        ),
        "no_action is not selectable.",
        "If you choose no action, set action to null.",
        "For medical treatment, choose family -> kind_hint -> params.",
        (
            "For medication-like actions, specify a non-empty params.drug_name. "
            "Dose and unit may be null. If the medication cannot be specified, "
            "set action to null, ask a focused question, order a diagnostic "
            "test, or choose a non-medication stabilization action. Do not "
            "invent drug_name if unsupported."
        ),
        "Params may be null if unknown or unspecified.",
        "Do not include raw_text.",
        "Do not include internal reasoning.",
        "Do not include chain-of-thought.",
        "Diagnostic order must choose exactly one available test name.",
        "Do not invent diagnostic results.",
        "Use released results only if present in observation.",
        "Output strict JSON only.",
    ]

    if runtime_input.turn_index is not None:
        sections.append(f"Turn index: {runtime_input.turn_index}")
    if runtime_input.profile is not None:
        sections.append(f"Profile:\n{_format_json(runtime_input.profile)}")
        if runtime_input.profile.traits:
            sections.append(
                f"Profile traits:\n{_format_json(runtime_input.profile.traits)}"
            )

    sections.extend(
        [
            f"Role-specific observation:\n{_format_json(runtime_input.observation)}",
            (
                "Available medical treatment action menu:\n"
                f"{_format_json(_build_action_menu(registry))}"
            ),
            (
                "Available diagnostic test names:\n"
                f"{_format_json(diagnostic_test_names)}"
            ),
            (
                "Expected JSON response shape:\n"
                "{\n"
                '  "verbal_action": {\n'
                '    "speaker": "clinician",\n'
                '    "target": "patient" | "nurse" | "relative" | null,\n'
                '    "content": "...",\n'
                '    "requires_response": true | false\n'
                "  } | null,\n"
                '  "action": {\n'
                '    "type": "medical_treatment_order",\n'
                '    "family": "...",\n'
                '    "kind_hint": "...",\n'
                '    "params": {...}\n'
                "  } | {\n"
                '    "type": "diagnostic_order",\n'
                '    "test_name": "..."\n'
                "  } | null\n"
                "}"
            ),
        ]
    )
    return "\n\n".join(sections)


def parse_clinician_response(output: str | Mapping[str, Any]) -> ClinicianProposal:
    """Parse final clinician JSON into the shared ClinicianProposal model."""

    data = _coerce_json_object(output)
    forbidden_key_path = _find_forbidden_key(data)
    if forbidden_key_path is not None:
        raise ClinicianParserError(
            "Clinician responses must not include raw_text, internal_reasoning, "
            f"or chain_of_thought fields: {forbidden_key_path}"
        )
    normalized = _normalize_clinician_response(data)
    try:
        return parse_agent_proposal(normalized, role="clinician")
    except ValueError as exc:
        raise ClinicianParserError(f"Invalid clinician response shape: {exc}") from exc


parse_clinician_proposal = parse_clinician_response


def _normalize_clinician_response(data: Mapping[str, Any]) -> dict[str, Any]:
    if "verbal_action" not in data or "action" not in data:
        raise ClinicianParserError(
            "Clinician response must include verbal_action and action fields."
        )

    alias_keys = sorted(TOP_LEVEL_ACTION_ALIASES.intersection(data))
    if alias_keys:
        raise ClinicianParserError(
            "Clinician response must represent at most one action using the "
            f"single top-level action field; found {alias_keys}."
        )
    extra_top_level_keys = sorted(set(data) - TOP_LEVEL_KEYS)
    if extra_top_level_keys:
        raise ClinicianParserError(
            "Clinician response must contain only verbal_action and action; "
            f"found extra keys {extra_top_level_keys}."
        )

    return {
        "verbal_action": _normalize_verbal_action(data["verbal_action"]),
        "action": _normalize_action(data["action"]),
    }


def _normalize_verbal_action(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ClinicianParserError("verbal_action must be an object or null.")

    verbal_action = dict(value)
    has_target = "target" in verbal_action
    has_recipient = "recipient" in verbal_action
    if has_target and has_recipient:
        raise ClinicianParserError(
            "verbal_action must not include both target and recipient."
        )
    if has_target:
        verbal_action["recipient"] = verbal_action.pop("target")
    return verbal_action


def _normalize_action(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        raise ClinicianParserError(
            "Clinician may produce at most one action; action must not be a list."
        )
    if not isinstance(value, Mapping):
        raise ClinicianParserError("action must be an object or null.")

    action = dict(value)
    action_type = action.get("type")
    if action_type == "no_action" or action.get("kind_hint") == "no_action":
        raise ClinicianParserError("no_action is not selectable by the clinician.")
    if action_type not in ACTION_TYPES:
        raise ClinicianParserError(
            "action.type must be medical_treatment_order, diagnostic_order, or null."
        )

    if action_type == "diagnostic_order":
        extra_keys = sorted(set(action) - DIAGNOSTIC_ORDER_KEYS)
        if extra_keys:
            raise ClinicianParserError(
                "diagnostic_order must contain only type and test_name; "
                f"found extra keys {extra_keys}."
            )
        test_name = action.get("test_name")
        if not isinstance(test_name, str) or not test_name:
            raise ClinicianParserError("diagnostic_order.test_name is required.")
        return {"type": "diagnostic_order", "test_name": test_name}

    extra_keys = sorted(set(action) - MEDICAL_TREATMENT_ORDER_KEYS)
    if extra_keys:
        raise ClinicianParserError(
            "medical_treatment_order must contain only type, family, kind_hint, "
            f"and params; found extra keys {extra_keys}."
        )
    missing_keys = sorted(MEDICAL_TREATMENT_ORDER_KEYS - set(action))
    if missing_keys:
        raise ClinicianParserError(
            "medical_treatment_order is missing required shape keys "
            f"{missing_keys}."
        )
    params = action.get("params")
    if params is not None and not isinstance(params, Mapping):
        raise ClinicianParserError(
            "medical_treatment_order.params must be an object or null if present."
        )
    return action


def _coerce_json_object(output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            data = json.loads(
                output,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ClinicianParserError(
                "Clinician response must be valid strict JSON."
            ) from exc
    elif isinstance(output, Mapping):
        data = dict(output)
    else:
        raise ClinicianParserError("Clinician response must be a JSON object.")

    if not isinstance(data, dict):
        raise ClinicianParserError("Clinician response must be a JSON object.")
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


def _build_action_menu(registry: ActionRegistry) -> list[dict[str, Any]]:
    menu: list[dict[str, Any]] = []
    for family in registry.list_families():
        kind_entries = []
        for kind_hint in registry.list_kind_hints_for_family(family):
            definition = registry.get_definition(kind_hint)
            if not definition.clinician_selectable:
                continue
            kind_entries.append(
                {
                    "kind_hint": kind_hint,
                    "params_template": registry.get_params_template(kind_hint),
                    "param_options": registry.get_param_options(kind_hint),
                    "param_guidance": registry.get_param_guidance(kind_hint),
                }
            )
        if kind_entries:
            menu.append({"family": family, "kind_hints": kind_entries})
    return menu


def _extract_available_diagnostic_test_names(
    observation: Mapping[str, Any],
) -> list[str]:
    for key in (
        "available_diagnostic_tests",
        "available_diagnostic_test_names",
        "diagnostic_test_names",
    ):
        names = _coerce_test_name_list(observation.get(key))
        if names:
            return names
    return []


def _coerce_test_name_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _coerce_profile(
    profile: AgentProfile | Mapping[str, Any] | None,
) -> AgentProfile | None:
    if profile is None:
        return None
    if isinstance(profile, AgentProfile):
        if profile.role != "clinician":
            raise ValueError(f"Clinician profile requires role='clinician'.")
        return profile
    coerced = AgentProfile.model_validate(profile)
    if coerced.role != "clinician":
        raise ValueError(f"Clinician profile requires role='clinician'.")
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
    "ClinicianAgent",
    "ClinicianParserError",
    "LLMCallable",
    "build_clinician_prompt",
    "parse_clinician_proposal",
    "parse_clinician_response",
]
