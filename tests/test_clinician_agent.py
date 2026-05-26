from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.agents.clinician as clinician_module
from ed_world_model.actions.registry import ActionFamily, ActionRegistry, KindHint
from ed_world_model.actions.validator import ActionValidator
from ed_world_model.agents.clinician import (
    ClinicianAgent,
    ClinicianParserError,
    build_clinician_prompt,
    parse_clinician_response,
)
from ed_world_model.agents.schemas import AgentProfile, ClinicianProposal
from ed_world_model.state.global_state import (
    GlobalState,
    TestBankItem as DiagnosticTestBankItem,
)


def _observation() -> dict:
    return {
        "patient_state": {
            "vitals": {"HR": 120, "O2Sat": 88},
            "features": {"work_of_breathing": "increased"},
        },
        "known_facts": {"chief_complaint": "shortness of breath"},
        "available_diagnostic_tests": ["ECG", "CXR"],
        "newly_available_results": [{"name": "VBG", "result": "pH 7.31"}],
        "recent_messages": [
            {
                "speaker": "patient",
                "recipient": "clinician",
                "content": "I cannot breathe.",
            }
        ],
        "pending_questions": [],
        "last_turn_events": [],
    }


def _prompt() -> str:
    return build_clinician_prompt(
        _observation(),
        profile=AgentProfile(
            role="clinician",
            name="Dr. Lee",
            traits={"experience_level": "attending"},
        ),
        action_registry=ActionRegistry(),
    )


def _json_output(verbal_action=None, action=None) -> str:
    return json.dumps({"verbal_action": verbal_action, "action": action})


def test_build_prompt_includes_clinician_role() -> None:
    prompt = _prompt()

    assert "You are the clinician in an ED simulation." in prompt
    assert '"role": "clinician"' in prompt


def test_build_prompt_includes_partial_observation() -> None:
    prompt = _prompt()

    assert "partial role-specific observation" in prompt
    assert "Do not assume hidden state not shown" in prompt
    assert "shortness of breath" in prompt
    assert "I cannot breathe." in prompt


def test_build_prompt_includes_action_families_and_kind_hint_menu() -> None:
    prompt = _prompt()

    assert ActionFamily.RESPIRATORY_SUPPORT in prompt
    assert KindHint.OXYGEN_SUPPORT in prompt
    assert "params_template" in prompt
    assert "param_guidance" in prompt


def test_build_prompt_includes_available_diagnostic_test_names() -> None:
    prompt = _prompt()

    assert "Available diagnostic test names" in prompt
    assert "ECG" in prompt
    assert "CXR" in prompt


def test_build_prompt_says_at_most_one_structured_action_per_turn() -> None:
    assert "at most one structured action per turn" in _prompt()


def test_build_prompt_says_verbal_action_matches_single_structured_action() -> None:
    prompt = _prompt()

    assert (
        "verbal_action must be consistent with the single structured action"
        in prompt
    )
    assert "for this turn" in prompt


def test_build_prompt_says_not_to_mention_unstructured_extra_care_steps() -> None:
    prompt = _prompt()

    assert "must not promise, order, prepare, or describe" in prompt
    assert "additional tests or treatments" in prompt
    assert "not represented by the structured action" in prompt


def test_build_prompt_disallows_multi_action_verbal_pattern() -> None:
    prompt = _prompt()

    assert '"start oxygen and prepare ECG and chest X-ray"' in prompt
    assert "when only one structured action is allowed" in prompt
    assert "choose the single most important action for this turn" in prompt


def test_build_prompt_says_no_action_is_not_selectable() -> None:
    assert "no_action is not selectable" in _prompt()


def test_build_prompt_says_raw_text_must_not_be_included() -> None:
    assert "Do not include raw_text." in _prompt()


def test_build_prompt_says_internal_reasoning_and_chain_of_thought_are_forbidden() -> None:
    prompt = _prompt()

    assert "Do not include internal reasoning." in prompt
    assert "Do not include chain-of-thought." in prompt
    assert "If you produce a verbal_action" in prompt
    assert "silently reason about what the clinician would say" in prompt
    assert "who they would address" in prompt
    assert "If you produce a clinician action" in prompt
    assert "silently reason about which single medical_treatment_order" in prompt
    assert "Avoid repeating the same information" in prompt
    assert "Repetition is allowed when you are directly asked again" in prompt
    assert "correcting a misunderstanding" in prompt
    assert "new clinical or conversation context makes repetition necessary" in prompt
    assert "Do not output this reasoning" in prompt


def test_build_prompt_includes_emergency_stabilization_rule() -> None:
    prompt = _prompt()

    assert "Emergency stabilization" in prompt
    assert "immediately dangerous airway/breathing/circulation findings" in prompt
    assert "urgent stabilization first" in prompt
    assert "Do not delay life-saving care for a long history checklist." in prompt


def test_build_prompt_says_ask_patient_directly_if_conscious_and_can_speak() -> None:
    prompt = _prompt()

    assert "If the patient is conscious and can_speak=True" in prompt
    assert "ask one brief focused question" in prompt
    assert "Prefer asking the patient directly for symptoms/onset." in prompt
    assert "Early patient contact" in prompt
    assert "strongly prefer a patient-directed verbal_action" in prompt
    assert "requires_response=true" in prompt


def test_build_prompt_allows_patient_question_with_urgent_action() -> None:
    prompt = _prompt()

    assert "A focused history question is not an additional clinical action." in prompt
    assert "single verbal_action even when the structured action is an urgent treatment" in prompt
    assert "diagnostic order" in prompt
    assert "Pair the urgent action with one brief patient question" in prompt


def test_build_prompt_says_ask_at_most_one_focused_question_per_turn() -> None:
    prompt = _prompt()

    assert "One-question limit" in prompt
    assert "Ask at most one focused history question per turn." in prompt
    assert "Do not ask a long checklist in one message." in prompt


def test_build_prompt_includes_key_missing_history_facts() -> None:
    prompt = _prompt()

    assert "chief complaint" in prompt
    assert "symptoms/onset/progression" in prompt
    assert "allergies" in prompt
    assert "PMH" in prompt
    assert "home medications" in prompt


def test_build_prompt_warns_against_information_dependent_treatment_without_context() -> None:
    prompt = _prompt()

    assert "Information-dependent treatment" in prompt
    assert "Before diagnosis-specific or medication-heavy treatments" in prompt
    assert "consider whether enough key facts are known" in prompt
    assert "ask one focused question instead of guessing" in prompt


def test_build_prompt_allows_focused_question_in_treatment_verbal_action() -> None:
    prompt = _prompt()

    assert "If action.type is medical_treatment_order" in prompt
    assert "ask one focused patient history question relevant to the current problem" in prompt
    assert "If action.type is diagnostic_order" in prompt


def test_build_prompt_includes_pending_available_diagnostic_result_rule() -> None:
    prompt = _prompt()

    assert "Diagnostic result use" in prompt
    assert "Before ordering a diagnostic test" in prompt
    assert "check pending diagnostic results and available diagnostic results" in prompt
    assert "known_facts.available_results" in prompt
    assert "newly_available_results" in prompt


def test_build_prompt_includes_exact_diagnostic_result_grounding_rule() -> None:
    prompt = _prompt()

    assert "Diagnostic result grounding" in prompt
    assert "use only the exact test names and results present" in prompt
    assert "current observation" in prompt
    assert "refer to it by its exact displayed test name" in prompt


def test_build_prompt_says_not_to_substitute_or_infer_different_test_name() -> None:
    prompt = _prompt()

    assert "Do not infer or substitute a different test name." in prompt
    assert "available result belongs to a different test" in prompt


def test_build_prompt_says_pending_tests_are_pending_not_available() -> None:
    prompt = _prompt()

    assert "If a diagnostic test is pending" in prompt
    assert "describe it as pending/in progress rather than available" in prompt


def test_build_prompt_says_not_to_claim_result_ready_unless_observed() -> None:
    prompt = _prompt()

    assert "If a diagnostic result is not present in the observation" in prompt
    assert "do not claim it is ready or available" in prompt


def test_build_prompt_says_not_to_order_duplicate_pending_tests() -> None:
    prompt = _prompt()

    assert "Do not request or order a diagnostic test that is already pending." in prompt
    assert "refer to it as pending or in progress rather than ordering it again" in prompt


def test_build_prompt_says_not_to_order_tests_with_available_results() -> None:
    prompt = _prompt()

    assert (
        "Do not request or order a diagnostic test whose result is already available."
        in prompt
    )


def test_build_prompt_says_use_available_result_instead_of_reordering() -> None:
    prompt = _prompt()

    assert "If a diagnostic result is available" in prompt
    assert "use it in your reasoning or communication" in prompt
    assert "If a result is already available" in prompt
    assert "use the available result instead of ordering the same test again" in prompt
    assert "instead of ordering the same test again" in prompt
    assert "verbal_action must not claim a duplicate test is being ordered" in prompt


def test_build_prompt_includes_numeric_fidelity_rule() -> None:
    prompt = _prompt()

    assert "Numeric vital fidelity" in prompt
    assert "use the values exactly as shown in the observation" in prompt
    assert "use the exact values shown in the observation" in prompt
    assert "low, high, hypotensive, hypertensive, tachycardic" in prompt
    assert "If unsure, state the numeric value without interpreting it." in prompt
    assert "If uncertain, state the numeric value without interpreting it." in prompt
    assert "must be consistent with the observed vitals and known facts" in prompt
    assert "pending results, and available results" in prompt


def test_build_prompt_says_not_to_invent_unsupported_rationale() -> None:
    prompt = _prompt()

    assert "rationale grounding" in prompt
    assert "make the verbal reason consistent with the observed vitals" in prompt
    assert "known facts, pending results, and available results" in prompt
    assert "Do not invent a rationale that is not supported by the observation." in prompt


def test_build_prompt_includes_deterioration_response_rule() -> None:
    prompt = _prompt()

    assert "Deterioration response" in prompt
    assert "remains unstable or is worsening despite prior interventions" in prompt
    assert "prioritize reassessment and stabilizing or escalating actions" in prompt
    assert "over non-urgent history questions" in prompt
    assert "Do not continue asking non-urgent history questions" in prompt
    assert "unless the information is immediately necessary for the next action" in prompt


def test_build_prompt_does_not_add_vasopressor_specific_hard_rule() -> None:
    prompt = _prompt().lower()

    assert "do not order vasopressor" not in prompt
    assert "vasopressor unless" not in prompt
    assert "vasopressor-specific" not in prompt


def test_build_prompt_does_not_add_vasopressor_fluid_oxygen_specific_hard_rules() -> None:
    prompt = _prompt().lower()

    for action_name in ("vasopressor", "fluid", "oxygen"):
        assert f"do not order {action_name}" not in prompt
        assert f"{action_name} unless" not in prompt
        assert f"{action_name}-specific" not in prompt


def test_build_prompt_does_not_include_scenario_specific_test_mismatch_examples() -> None:
    prompt = _prompt()

    assert "ECG result is available when" not in prompt
    assert "CXR result is available when" not in prompt
    assert "VBG result is available when" not in prompt
    assert "for example, ECG" not in prompt


def test_build_prompt_does_not_add_family_or_kind_hint_clinical_guidance() -> None:
    prompt = _prompt().lower()

    assert "family-level clinical guidance" not in prompt
    assert "kind_hint-level clinical guidance" not in prompt
    assert "clinical indication" not in prompt
    assert "clinical contraindication" not in prompt


def test_parse_valid_verbal_only_response() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action={
                "speaker": "clinician",
                "target": "patient",
                "content": "I am going to help your breathing.",
                "requires_response": False,
            },
            action=None,
        )
    )

    assert isinstance(proposal, ClinicianProposal)
    assert proposal.action is None
    assert proposal.verbal_action is not None
    assert proposal.verbal_action.recipient == "patient"


def test_parse_valid_medical_treatment_order_response() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action=None,
            action={
                "type": "medical_treatment_order",
                "family": ActionFamily.RESPIRATORY_SUPPORT,
                "kind_hint": KindHint.OXYGEN_SUPPORT,
                "params": {"oxygen_device": "NRB", "FiO2": 1.0},
            },
        )
    )

    assert proposal.verbal_action is None
    assert proposal.action["type"] == "medical_treatment_order"
    assert proposal.action["family"] == ActionFamily.RESPIRATORY_SUPPORT
    assert proposal.action["kind_hint"] == KindHint.OXYGEN_SUPPORT


def test_parse_valid_diagnostic_order_response() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action=None,
            action={"type": "diagnostic_order", "test_name": "ECG"},
        )
    )

    assert proposal.action == {"type": "diagnostic_order", "test_name": "ECG"}


def test_parse_valid_verbal_and_action_response() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action={
                "speaker": "clinician",
                "target": "nurse",
                "content": "Please place a non-rebreather.",
                "requires_response": False,
            },
            action={
                "type": "medical_treatment_order",
                "family": ActionFamily.RESPIRATORY_SUPPORT,
                "kind_hint": KindHint.OXYGEN_SUPPORT,
                "params": {"oxygen_device": "NRB"},
            },
        )
    )

    assert proposal.verbal_action.recipient == "nurse"
    assert proposal.action["kind_hint"] == KindHint.OXYGEN_SUPPORT


def test_parse_multi_action_verbal_text_still_parses_without_semantic_rejection() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action={
                "speaker": "clinician",
                "target": "patient",
                "content": "We will start oxygen and prepare ECG and chest X-ray.",
                "requires_response": False,
            },
            action={
                "type": "medical_treatment_order",
                "family": ActionFamily.RESPIRATORY_SUPPORT,
                "kind_hint": KindHint.OXYGEN_SUPPORT,
                "params": {"oxygen_device": "NRB"},
            },
        )
    )

    assert proposal.verbal_action.content == (
        "We will start oxygen and prepare ECG and chest X-ray."
    )
    assert proposal.action["kind_hint"] == KindHint.OXYGEN_SUPPORT


def test_parse_null_action_as_no_clinician_action() -> None:
    proposal = parse_clinician_response(
        _json_output(verbal_action=None, action=None)
    )

    assert proposal.verbal_action is None
    assert proposal.action is None


def test_reject_no_action_action_type() -> None:
    with pytest.raises(ClinicianParserError, match="no_action"):
        parse_clinician_response(
            _json_output(verbal_action=None, action={"type": "no_action"})
        )

    with pytest.raises(ClinicianParserError, match="no_action"):
        parse_clinician_response(
            _json_output(
                verbal_action=None,
                action={
                    "type": "medical_treatment_order",
                    "family": ActionFamily.TIME_PROGRESSION,
                    "kind_hint": KindHint.NO_ACTION,
                    "params": {"elapsed_min": 1},
                },
            )
        )


def test_reject_more_than_one_action_if_represented() -> None:
    with pytest.raises(ClinicianParserError, match="at most one action"):
        parse_clinician_response(
            {
                "verbal_action": None,
                "action": [{"type": "diagnostic_order", "test_name": "ECG"}],
            }
        )

    with pytest.raises(ClinicianParserError, match="at most one action"):
        parse_clinician_response(
            {
                "verbal_action": None,
                "action": {"type": "diagnostic_order", "test_name": "ECG"},
                "medical_treatment_order": {"kind_hint": "oxygen_support"},
            }
        )


def test_reject_raw_text_anywhere() -> None:
    with pytest.raises(ClinicianParserError, match="raw_text"):
        parse_clinician_response(
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "target": "patient",
                    "content": "I am here.",
                    "requires_response": False,
                },
                "action": {
                    "type": "medical_treatment_order",
                    "family": ActionFamily.RESPIRATORY_SUPPORT,
                    "kind_hint": KindHint.OXYGEN_SUPPORT,
                    "params": {"raw_text": "legacy action text"},
                },
            }
        )


def test_reject_internal_reasoning_and_chain_of_thought_anywhere() -> None:
    with pytest.raises(ClinicianParserError, match="internal_reasoning"):
        parse_clinician_response(
            {
                "internal_reasoning": "hidden deliberation",
                "verbal_action": None,
                "action": None,
            }
        )

    with pytest.raises(ClinicianParserError, match="chain_of_thought"):
        parse_clinician_response(
            {
                "verbal_action": None,
                "action": {
                    "type": "diagnostic_order",
                    "test_name": "ECG",
                    "chain_of_thought": "...",
                },
            }
        )


def test_reject_invalid_json() -> None:
    with pytest.raises(ClinicianParserError, match="valid strict JSON"):
        parse_clinician_response('{"verbal_action": null, "action": null')


def test_reject_invalid_output_shape_extra_fields() -> None:
    with pytest.raises(ClinicianParserError, match="only verbal_action and action"):
        parse_clinician_response(
            {"verbal_action": None, "action": None, "confidence": 0.75}
        )

    with pytest.raises(ClinicianParserError, match="medical_treatment_order"):
        parse_clinician_response(
            {
                "verbal_action": None,
                "action": {
                    "type": "medical_treatment_order",
                    "family": ActionFamily.RESPIRATORY_SUPPORT,
                    "kind_hint": KindHint.OXYGEN_SUPPORT,
                    "params": {},
                    "rationale": "not part of final proposal shape",
                },
            }
        )


def test_preserve_null_params() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action=None,
            action={
                "type": "medical_treatment_order",
                "family": ActionFamily.RESPIRATORY_SUPPORT,
                "kind_hint": KindHint.OXYGEN_SUPPORT,
                "params": None,
            },
        )
    )

    assert proposal.action["params"] is None


def test_diagnostic_order_parsed_output_contains_test_name_only_not_result() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action=None,
            action={"type": "diagnostic_order", "test_name": "ECG"},
        )
    )

    assert proposal.action == {"type": "diagnostic_order", "test_name": "ECG"}

    with pytest.raises(ClinicianParserError, match="only type and test_name"):
        parse_clinician_response(
            _json_output(
                verbal_action=None,
                action={
                    "type": "diagnostic_order",
                    "test_name": "ECG",
                    "result": "STEMI",
                },
            )
        )


def test_fake_llm_callable_generate_path_returns_parsed_proposal() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return _json_output(
            verbal_action={
                "speaker": "clinician",
                "target": "patient",
                "content": "We are checking your heart tracing.",
                "requires_response": False,
            },
            action={"type": "diagnostic_order", "test_name": "ECG"},
        )

    agent = ClinicianAgent(
        fake_llm,
        profile=AgentProfile(role="clinician", traits={"style": "concise"}),
    )

    proposal = agent.generate(_observation())

    assert len(prompts) == 1
    assert "You are the clinician in an ED simulation." in prompts[0]
    assert proposal.verbal_action.recipient == "patient"
    assert proposal.action == {"type": "diagnostic_order", "test_name": "ECG"}


def test_no_real_llm_api_calls_or_imports() -> None:
    source = inspect.getsource(clinician_module)

    assert "openai" not in source.lower()
    assert "requests" not in source
    assert "urllib" not in source


def test_clinician_agent_parser_does_not_import_transition_engines() -> None:
    source = inspect.getsource(clinician_module)

    assert "transition_engines" not in source


def test_parsed_valid_medical_action_passes_action_validator() -> None:
    proposal = parse_clinician_response(
        _json_output(
            verbal_action=None,
            action={
                "type": "medical_treatment_order",
                "family": ActionFamily.RESPIRATORY_SUPPORT,
                "kind_hint": KindHint.OXYGEN_SUPPORT,
                "params": {"oxygen_device": "NRB", "FiO2": 1.0},
            },
        )
    )
    state = GlobalState(
        truth_state={
            "test_bank": [DiagnosticTestBankItem(name="ECG", result="sinus rhythm")]
        }
    )

    result = ActionValidator(ActionRegistry()).validate_clinician_proposal(
        proposal.model_dump(exclude_none=True),
        state,
    )

    assert result.ok is True
    assert result.action_type == "medical_treatment_order"
    assert result.normalized_action["raw_text"] is None
