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
