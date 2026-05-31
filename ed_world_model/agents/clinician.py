"""Clinician LLM prompt and parser boundary for v1.3.1."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    VerbalAction,
    VerbalDecision,
)
from ed_world_model.agents.verbal_decision import (
    VerbalDecisionParserError,
    parse_verbal_decision,
)


LLMCallable = Callable[[str], str]
ACTION_TYPES = {"medical_treatment_order", "diagnostic_order"}
TOP_LEVEL_ACTION_ALIASES = {"actions", "medical_treatment_order", "diagnostic_order"}
TOP_LEVEL_KEYS = {"verbal_action", "action"}
DECISION_TOP_LEVEL_KEYS = {"action", "verbal_decision"}
FINAL_VERBAL_RESPONSE_KEYS = {"verbal_action"}
DIAGNOSTIC_ORDER_KEYS = {"type", "test_name"}
MEDICAL_TREATMENT_ORDER_KEYS = {"type", "family", "kind_hint", "params"}
FINAL_CLINICIAN_VERBAL_FORBIDDEN_KEYS = {
    *FORBIDDEN_OUTPUT_KEYS,
    "action",
    "verbal_decision",
    "reasoning_summary",
}


class ClinicianParserError(ValueError):
    """Raised when a clinician LLM response is not valid final JSON output."""


@dataclass(frozen=True)
class ClinicianDecision:
    action: dict[str, Any] | None
    verbal_decision: VerbalDecision | None = None


class ClinicianAgent:
    """LLM-callable wrapper for the clinician role.

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
        action_registry: ActionRegistry | None = None,
    ) -> None:
        self._decision_llm_callable = decision_llm_callable or llm_callable
        self._verbal_llm_callable = verbal_llm_callable or llm_callable
        if self._decision_llm_callable is None or self._verbal_llm_callable is None:
            raise ValueError(
                "ClinicianAgent requires llm_callable or both decision and verbal callables."
            )
        self._profile = _coerce_profile(profile)
        self._action_registry = action_registry or ActionRegistry()
        self.last_verbal_decision: VerbalDecision | None = None

    def build_decision_prompt(self, observation: Mapping[str, Any]) -> str:
        return build_clinician_decision_prompt(
            observation,
            action_registry=self._action_registry,
        )

    def build_verbal_prompt(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any] | None,
        decision: VerbalDecision | Mapping[str, Any],
    ) -> str:
        return build_clinician_verbal_prompt(
            observation,
            action,
            decision,
            profile=self._profile,
        )

    def build_prompt(self, observation: Mapping[str, Any]) -> str:
        return self.build_decision_prompt(observation)

    def generate(self, observation: dict[str, Any]) -> ClinicianProposal:
        self.last_verbal_decision = None
        decision_prompt = self.build_decision_prompt(observation)
        decision_output = self._decision_llm_callable(decision_prompt)
        decision = parse_clinician_decision_response(decision_output)
        self.last_verbal_decision = decision.verbal_decision
        if decision.verbal_decision is None or not decision.verbal_decision.should_speak:
            return ClinicianProposal(verbal_action=None, action=decision.action)

        verbal_prompt = self.build_verbal_prompt(
            observation,
            decision.action,
            decision.verbal_decision,
        )
        verbal_output = self._verbal_llm_callable(verbal_prompt)
        verbal_action = parse_clinician_verbal_response(verbal_output)
        return ClinicianProposal(verbal_action=verbal_action, action=decision.action)


def build_clinician_decision_prompt(
    observation: Mapping[str, Any],
    *,
    action_registry: ActionRegistry | None = None,
    available_diagnostic_test_names: Sequence[str] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build the clinician Stage 1 action and VerbalDecision prompt."""

    registry = action_registry or ActionRegistry()
    decision_observation = _clinician_decision_observation(observation)
    diagnostic_test_names = list(
        available_diagnostic_test_names
        if available_diagnostic_test_names is not None
        else _extract_available_diagnostic_test_names(decision_observation)
    )

    sections = [
        "You are the clinician in an ED simulation.",
        "Stage 1: decide the structured action and verbal communication decision.",
        "Output structured JSON only.",
        "Do not output the final verbal message in Stage 1.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        (
            "Stage 1 does not use AgentProfile, traits, communication style, "
            "personality, or emotion_context. Do not use style or emotion to "
            "choose the clinical action or communication content."
        ),
        "Do not include chain-of-thought or internal reasoning.",
        (
            "Provide only a concise reasoning_summary in verbal_decision, "
            "grounded in the observation."
        ),
        (
            "Decide at most one action: one medical_treatment_order, one "
            "diagnostic_order, or null."
        ),
        "no_action is not selectable.",
        "For medical treatment, choose family -> kind_hint -> params.",
        (
            "For medication-like actions, specify a non-empty params.drug_name. "
            "Dose and unit may be null. Do not invent drug_name if unsupported."
        ),
        "Diagnostic order must choose exactly one available test name.",
        (
            "verbal_decision must be consistent with the selected action and "
            "must not mention tests, treatments, or actions not represented by "
            "the single structured action."
        ),
        (
            "If action is null, verbal_decision must not claim a treatment or "
            "test is being started, ordered, prepared, or performed."
        ),
        (
            "If a diagnostic result is referenced, use exact test names and "
            "results from the observation."
        ),
        (
            "If a diagnostic test is pending, describe it as pending/in "
            "progress rather than available."
        ),
        (
            "When referencing vital signs, use exact vital sign values shown "
            "in the observation."
        ),
        (
            "If the patient can speak and focused history is needed, "
            "verbal_decision may target patient with one focused question."
        ),
        (
            "If there is an unresolved pending question targeting patient, do "
            "not ask another patient question unless urgent and clearly different."
        ),
        (
            "Use key_points for facts, action details, or one question the "
            "final verbal message may include. Use forbidden_points for extra "
            "tests, treatments, results, facts, or unsupported rationale the "
            "final message must avoid."
        ),
        (
            "Expected JSON response shape:\n"
            "{\n"
            '  "action": {\n'
            '    "type": "medical_treatment_order",\n'
            '    "family": "...",\n'
            '    "kind_hint": "...",\n'
            '    "params": {...}\n'
            "  } | {\n"
            '    "type": "diagnostic_order",\n'
            '    "test_name": "..."\n'
            "  } | null,\n"
            '  "verbal_decision": {\n'
            '    "speaker": "clinician",\n'
            '    "should_speak": true | false,\n'
            '    "target": "clinician" | "nurse" | "patient" | "relative" | null,\n'
            '    "intent": "..." | null,\n'
            '    "reasoning_summary": "..." | null,\n'
            '    "key_points": ["..."],\n'
            '    "forbidden_points": ["..."],\n'
            '    "requires_response": true | false\n'
            "  } | null\n"
            "}"
        ),
    ]

    if turn_index is not None:
        sections.append(f"Turn index: {turn_index}")

    sections.extend(
        [
            f"Role-specific observation:\n{_format_json(decision_observation)}",
            (
                "Available medical treatment action menu:\n"
                f"{_format_json(_build_action_menu(registry))}"
            ),
            (
                "Available diagnostic test names:\n"
                f"{_format_json(diagnostic_test_names)}"
            ),
        ]
    )
    return "\n\n".join(sections)


def build_clinician_verbal_prompt(
    observation: Mapping[str, Any],
    action: Mapping[str, Any] | None,
    decision: VerbalDecision | Mapping[str, Any],
    *,
    profile: AgentProfile | Mapping[str, Any] | None = None,
    emotion_context: Mapping[str, Any] | None = None,
    turn_index: int | None = None,
) -> str:
    """Build the clinician Stage 2 final verbal-action prompt."""

    clinician_profile = _coerce_profile(profile)
    verbal_decision = _coerce_clinician_decision(decision)

    sections = [
        "You are the clinician in an ED simulation.",
        "Stage 2: generate final verbal_action JSON only.",
        COMMON_PARTIAL_OBSERVATION_RULE,
        "Use VerbalDecision and selected action as the only plan.",
        (
            "You may use AgentProfile, traits, and emotion context only to "
            "shape wording and tone."
        ),
        (
            "Do not let profile, traits, or emotion change observed facts, "
            "selected action, available results, or grounded clinical content."
        ),
        "Do not include VerbalDecision.",
        "Do not include reasoning_summary.",
        "Do not include chain-of-thought or internal reasoning.",
        "Do not mention extra actions not in the structured action.",
        "Do not add diagnostic results or facts not present in the observation.",
        (
            "If action is null, do not claim that a treatment or test is being "
            "started, ordered, prepared, or performed."
        ),
        (
            "If the final message references a diagnostic result, use exact "
            "test names and results from the observation."
        ),
        (
            "If the final message references a pending test, call it pending "
            "or in progress."
        ),
        (
            "If the final message references vital signs, use exact vital sign "
            "values from the observation."
        ),
        (
            "Set verbal_action.requires_response=true only if the final message "
            "explicitly asks a question."
        ),
        (
            "Expected JSON response shape:\n"
            "{\n"
            '  "verbal_action": {\n'
            '    "speaker": "clinician",\n'
            '    "target": "patient" | "nurse" | "relative" | null,\n'
            '    "content": "...",\n'
            '    "requires_response": true | false\n'
            "  } | null\n"
            "}"
        ),
    ]

    if turn_index is not None:
        sections.append(f"Turn index: {turn_index}")
    if clinician_profile is not None:
        sections.append(f"Profile:\n{_format_json(clinician_profile)}")
        if clinician_profile.traits:
            sections.append(
                f"Profile traits:\n{_format_json(clinician_profile.traits)}"
            )
    if emotion_context is not None:
        sections.append(f"Emotion context:\n{_format_json(emotion_context)}")

    sections.extend(
        [
            f"Selected action:\n{_format_json(action)}",
            f"VerbalDecision:\n{_format_json(verbal_decision)}",
            f"Role-specific observation:\n{_format_json(observation)}",
        ]
    )
    return "\n\n".join(sections)


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


def parse_clinician_decision_response(
    output: str | Mapping[str, Any],
) -> ClinicianDecision:
    """Parse clinician Stage 1 JSON into action plus transient VerbalDecision."""

    data = _coerce_json_object(output)
    forbidden_key_path = _find_forbidden_key(data)
    if forbidden_key_path is not None:
        raise ClinicianParserError(
            "Clinician decision responses must not include raw_text, "
            "internal_reasoning, chain_of_thought, or intent_type fields: "
            f"{forbidden_key_path}"
        )
    extra_top_level_keys = sorted(set(data) - DECISION_TOP_LEVEL_KEYS)
    if extra_top_level_keys:
        raise ClinicianParserError(
            "Clinician decision response must contain only action and "
            f"verbal_decision; found extra keys {extra_top_level_keys}."
        )
    if "action" not in data or "verbal_decision" not in data:
        raise ClinicianParserError(
            "Clinician decision response must include action and verbal_decision."
        )

    action = _normalize_action(data["action"])
    verbal_decision = data["verbal_decision"]
    if verbal_decision is not None:
        try:
            parsed_decision = parse_verbal_decision(
                verbal_decision,
                speaker="clinician",
            )
        except VerbalDecisionParserError as exc:
            raise ClinicianParserError(
                f"Invalid clinician verbal_decision: {exc}"
            ) from exc
    else:
        parsed_decision = None
    return ClinicianDecision(action=action, verbal_decision=parsed_decision)


def parse_clinician_verbal_response(
    output: str | Mapping[str, Any],
) -> VerbalAction | None:
    """Parse clinician Stage 2 final verbal-action JSON."""

    data = _coerce_json_object(output)
    forbidden_key_path = _find_final_verbal_forbidden_key(data)
    if forbidden_key_path is not None:
        raise ClinicianParserError(
            "Clinician final verbal responses must not include action, "
            "verbal_decision, reasoning_summary, raw_text, internal_reasoning, "
            f"chain_of_thought, or intent_type fields: {forbidden_key_path}"
        )
    extra_top_level_keys = sorted(set(data) - FINAL_VERBAL_RESPONSE_KEYS)
    if extra_top_level_keys:
        raise ClinicianParserError(
            "Clinician final verbal response must contain only verbal_action; "
            f"found extra keys {extra_top_level_keys}."
        )
    if "verbal_action" not in data:
        raise ClinicianParserError(
            "Clinician final verbal response must include verbal_action."
        )
    verbal_action = _normalize_final_clinician_verbal_action(data["verbal_action"])
    if verbal_action is None:
        return None
    try:
        return VerbalAction.model_validate(verbal_action)
    except ValueError as exc:
        raise ClinicianParserError(
            f"Invalid clinician final verbal_action shape: {exc}"
        ) from exc


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


def _normalize_final_clinician_verbal_action(value: Any) -> dict[str, Any] | None:
    verbal_action = _normalize_verbal_action(value)
    if verbal_action is None:
        return None
    target = verbal_action.get("recipient")
    if target not in {"patient", "nurse", "relative", None}:
        raise ClinicianParserError(
            "verbal_action.target must be patient, nurse, relative, or null."
        )
    speaker = verbal_action.get("speaker")
    if speaker != "clinician":
        raise ClinicianParserError("verbal_action.speaker must be 'clinician'.")
    requires_response = verbal_action.get("requires_response")
    if (
        requires_response not in (False, None)
        and not _is_explicit_targeted_question(verbal_action.get("content"), target)
    ):
        verbal_action["requires_response"] = False
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


def _find_final_verbal_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = f"{path}.{key_text}"
            if key_text in FINAL_CLINICIAN_VERBAL_FORBIDDEN_KEYS:
                return current_path
            nested = _find_final_verbal_forbidden_key(item, current_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_final_verbal_forbidden_key(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _is_explicit_targeted_question(content: Any, target: Any) -> bool:
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


def _coerce_clinician_decision(
    decision: VerbalDecision | Mapping[str, Any],
) -> VerbalDecision:
    if isinstance(decision, VerbalDecision):
        if decision.speaker != "clinician":
            raise ValueError("Clinician VerbalDecision requires speaker='clinician'.")
        return decision
    coerced = VerbalDecision.model_validate(decision)
    if coerced.speaker != "clinician":
        raise ValueError("Clinician VerbalDecision requires speaker='clinician'.")
    return coerced


_CLINICIAN_DECISION_EXCLUDED_KEYS = {
    "agent_profile",
    "communication_style",
    "emotion_context",
    "patient_emotion",
    "patient_internal_state",
    "profile",
    "traits",
    "truth_state",
}


def _clinician_decision_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_keys_for_decision(observation, _CLINICIAN_DECISION_EXCLUDED_KEYS)


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
    "ClinicianDecision",
    "ClinicianParserError",
    "LLMCallable",
    "build_clinician_decision_prompt",
    "build_clinician_prompt",
    "build_clinician_verbal_prompt",
    "parse_clinician_decision_response",
    "parse_clinician_proposal",
    "parse_clinician_response",
    "parse_clinician_verbal_response",
]
