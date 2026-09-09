"""Clinician LLM prompt and parser boundary for v1.3.1."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ed_world_model.actions.registry import ActionRegistry
from ed_world_model.agents.prompting import (
    ANTI_REPETITION_RULE,
    CLINICAL_VERBAL_STYLE_NOT_TRAITS,
    CLINICAL_VERBAL_STYLE_POLICY,
    COMMON_PARTIAL_OBSERVATION_RULE,
    FORBIDDEN_OUTPUT_KEYS,
    build_json_repair_prompt,
    json_failure_snippet,
    strip_markdown_fence,
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
CLINICIAN_TO_NURSE_ORDER_POLICY = (
    "Clinician-to-nurse order communication: keep team-directed orders natural "
    "and concise. Use one order plus one short reason if needed. Do not stack "
    "multiple vitals, labs, or pathophysiology details unless they are essential "
    "to the immediate order. Use team communication, not a lab-value dump."
)
_CLINICIAN_DECISION_REPAIR_SHAPE = """{
  "action": {
    "type": "medical_treatment_order",
    "family": "...",
    "kind_hint": "...",
    "params": {}
  } | {
    "type": "diagnostic_order",
    "test_name": "..."
  } | null,
  "verbal_decision": {
    "speaker": "clinician",
    "should_speak": true | false,
    "target": "clinician" | "nurse" | "patient" | "relative" | null,
    "intent": "..." | null,
    "reasoning_summary": "..." | null,
    "key_points": ["..."],
    "forbidden_points": ["..."],
    "requires_response": true | false
  } | null
}"""
_CLINICIAN_VERBAL_REPAIR_SHAPE = """{
  "verbal_action": {
    "speaker": "clinician",
    "target": "patient" | "nurse" | "relative" | null,
    "content": "...",
    "requires_response": true | false
  } | null
}"""


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
        decision = self._parse_decision_with_repair(
            decision_prompt,
            decision_output,
        )
        self.last_verbal_decision = decision.verbal_decision
        if decision.verbal_decision is None or not decision.verbal_decision.should_speak:
            return ClinicianProposal(verbal_action=None, action=decision.action)

        verbal_prompt = self.build_verbal_prompt(
            observation,
            decision.action,
            decision.verbal_decision,
        )
        verbal_output = self._verbal_llm_callable(verbal_prompt)
        verbal_action = self._parse_verbal_with_repair(
            verbal_prompt,
            verbal_output,
        )
        return ClinicianProposal(verbal_action=verbal_action, action=decision.action)

    def _parse_decision_with_repair(
        self,
        prompt: str,
        output: str | Mapping[str, Any],
    ) -> ClinicianDecision:
        try:
            return parse_clinician_decision_response(output)
        except ClinicianParserError as first_error:
            repair_prompt = build_json_repair_prompt(
                stage="clinician Stage 1 decision",
                original_prompt=prompt,
                invalid_output=output,
                error_message=str(first_error),
                expected_shape=_CLINICIAN_DECISION_REPAIR_SHAPE,
            )
            repair_output = self._decision_llm_callable(repair_prompt)
            try:
                repaired = parse_clinician_decision_response(repair_output)
            except ClinicianParserError as repair_error:
                raise ClinicianParserError(
                    f"{first_error}; repair retry failed: {repair_error}; "
                    f"raw={json_failure_snippet(output)}; "
                    f"repair_raw={json_failure_snippet(repair_output)}"
                ) from first_error
            if _is_empty_decision(repaired):
                raise first_error
            return repaired

    def _parse_verbal_with_repair(
        self,
        prompt: str,
        output: str | Mapping[str, Any],
    ) -> VerbalAction | None:
        try:
            return parse_clinician_verbal_response(output)
        except ClinicianParserError as first_error:
            repair_prompt = build_json_repair_prompt(
                stage="clinician Stage 2 final verbal_action",
                original_prompt=prompt,
                invalid_output=output,
                error_message=str(first_error),
                expected_shape=_CLINICIAN_VERBAL_REPAIR_SHAPE,
            )
            repair_output = self._verbal_llm_callable(repair_prompt)
            try:
                return parse_clinician_verbal_response(repair_output)
            except ClinicianParserError as repair_error:
                raise ClinicianParserError(
                    f"{first_error}; repair retry failed: {repair_error}; "
                    f"raw={json_failure_snippet(output)}; "
                    f"repair_raw={json_failure_snippet(repair_output)}"
                ) from first_error


def _is_empty_decision(decision: ClinicianDecision) -> bool:
    return decision.action is None and (
        decision.verbal_decision is None
        or not decision.verbal_decision.should_speak
    )


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
            "or personality. Do not use style to "
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
        (
            "While the patient is not yet stabilized (abnormal vitals, ongoing "
            "distress, or an unresolved problem), do not go idle. Each turn, "
            "either take the next appropriate clinical action (escalate or "
            "adjust treatment, or order a needed test) or speak to the patient "
            "to explain, disclose, instruct, or gather focused history. Set action=null "
            "and should_speak=false only when the patient is stable and there "
            "is genuinely nothing to do or say."
        ),
        CLINICAL_VERBAL_STYLE_POLICY,
        CLINICAL_VERBAL_STYLE_NOT_TRAITS,
        CLINICIAN_TO_NURSE_ORDER_POLICY,
        (
            "administered_treatments in the observation is the complete log of "
            "treatments already executed this encounter. Never claim a "
            "treatment has not been given when it appears there. Ordering a "
            "treatment already in the log is allowed only as a deliberate, "
            "clinically justified repeat dose (e.g. repeat epinephrine in "
            "refractory anaphylaxis) - and state explicitly that it is a "
            "repeat."
        ),
        (
            "If the current treatment is not working (e.g., persistent hypoxia "
            "despite oxygen or BiPAP), escalate (e.g., prepare for intubation) "
            "instead of repeating the same order or going idle."
        ),
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
            "When referencing vitals, use the observation's values but phrase "
            "them naturally and rounded (e.g. 'your oxygen is 80%'), only when "
            "relevant to the listener. Do not recite the full vital-sign set "
            "as telemetry."
        ),
        (
            "Prefer addressing the patient directly (target=patient) to explain "
            "what you are doing, disclose clinically relevant information, give "
            "a brief instruction, or ask one focused history question "
            "(requires_response=true for a question). Use the nurse for orders, "
            "but do not leave the patient unaddressed for multiple turns."
        ),
        (
            "Patient question priority: When the patient is alive, conscious, "
            "and can_speak=True, and there is no unresolved pending question "
            "for the patient, strongly prefer using the verbal_decision to ask "
            "one focused patient question if key facts are missing. Prioritize "
            "current symptoms and onset/progression first, then allergies, "
            "past medical/surgical history, and home medications. Use "
            "target=patient and requires_response=true."
        ),
        (
            "Relative fallback: If the patient is not conscious or "
            "can_speak=False and key facts are missing, ask the relative one "
            "focused question about allergies, past medical/surgical history, "
            "home medications, baseline status, or recent course. Use "
            "target=relative and requires_response=true. Do not ask the nurse "
            "for subjective patient-owned history when a relative could answer."
        ),
        (
            "If there is an unresolved pending question targeting patient or "
            "relative, do not ask that person another question unless urgent "
            "and clearly different."
        ),
        (
            "Stop futile history-taking: If a source (the patient or the "
            "relative) has already answered that they do not know or cannot "
            "provide the information on two or more of your recent questions, "
            "treat that history as unavailable for this encounter. Stop asking "
            "that source history questions and do not keep re-asking different "
            "history items they are unlikely to know."
        ),
        (
            "Do not restate something you or another agent already said (for "
            "example 'she is intubated and cannot answer right now'). If your "
            "only available message would repeat prior content, re-ask a "
            "question already answered with 'I don't know', or add nothing new, "
            "set should_speak=false. Once the patient is stabilized and there "
            "is nothing new and useful to do or say, going idle (action=null, "
            "should_speak=false) is the correct choice -- silence is better than "
            "repetition."
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
        CLINICAL_VERBAL_STYLE_POLICY,
        CLINICAL_VERBAL_STYLE_NOT_TRAITS,
        CLINICIAN_TO_NURSE_ORDER_POLICY,
        (
            "Do not let profile metadata change observed facts, selected action, "
            "available results, or grounded clinical content."
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
            "If VerbalDecision targets patient or relative and its intent or "
            "key_points contain a focused history question, include exactly "
            "one short question in the final message and set "
            "requires_response=true. Do not turn the question into a statement."
        ),
        (
            "If the selected listener is relative because the patient cannot "
            "speak, ask the relative for only one missing key fact category "
            "such as allergies, past medical/surgical history, home medications, "
            "baseline status, or recent course."
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
        sections.append(
            "Profile:\n"
            f"{_format_json(clinician_profile.model_dump(exclude={'traits'}))}"
        )

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
            "the observation, recent messages, and this prompt's verbal style policy."
        ),
        CLINICAL_VERBAL_STYLE_POLICY,
        CLINICAL_VERBAL_STYLE_NOT_TRAITS,
        CLINICIAN_TO_NURSE_ORDER_POLICY,
        (
            "Clinical decisions must remain grounded in the role-specific observation, released "
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
            "especially early in the encounter and after immediate stabilization "
            "has begun. Prefer asking the patient directly for symptoms/onset. "
            "Also ask the patient for allergies, past medical/surgical history, "
            "and home medications when relevant. Ask the relative instead when "
            "the patient cannot answer."
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
            "Ongoing history gathering: After the first 1-2 turns, if the "
            "patient is still conscious and key facts remain missing, continue "
            "asking at most one focused question on turns where it does not "
            "delay urgent stabilization. Use a short direct question rather "
            "than a checklist."
        ),
        (
            "Post-intubation/failure-to-speak history: Once the patient cannot "
            "speak, stop asking patient-directed questions. If allergies, PMH, "
            "home medications, or recent course remain unknown, ask the "
            "relative one focused question with requires_response=true."
        ),
        (
            "Stop futile history-taking: If a source (the patient or the "
            "relative) has already answered that they do not know or cannot "
            "provide the information on two or more of your recent questions, "
            "treat that history as unavailable for this encounter. Stop asking "
            "that source history questions and do not keep re-asking different "
            "history items they are unlikely to know."
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
            "Do not restate something you or another agent already said (for "
            "example 'she is intubated and cannot answer right now'). If your "
            "only available message would repeat prior content, re-ask a "
            "question already answered with 'I don't know', or add nothing new, "
            "set should_speak=false. Once the patient is stabilized and there "
            "is nothing new and useful to do or say, going idle (action=null, "
            "should_speak=false) is the correct choice — silence is better than "
            "repetition."
        ),
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
        CLINICIAN_TO_NURSE_ORDER_POLICY,
        (
            "If action.type is medical_treatment_order, verbal_action may "
            "explain that treatment, give a brief instruction, disclose relevant "
            "clinical concern, provide reassurance only when appropriate, ask one "
            "focused patient history question relevant to the current problem, "
            "or if the patient cannot speak ask one focused relative history "
            "question relevant to the current problem."
        ),
        (
            "If action.type is diagnostic_order, verbal_action may explain only "
            "that one diagnostic test, give a brief instruction, disclose relevant "
            "clinical concern, provide reassurance only when appropriate, ask one "
            "focused patient history question relevant to the current problem, "
            "or if the patient cannot speak ask one focused relative history "
            "question relevant to the current problem."
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
        sections.append(
            "Profile:\n"
            f"{_format_json(runtime_input.profile.model_dump(exclude={'traits'}))}"
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
                strip_markdown_fence(output),
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
