from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.agents.nurse as nurse_module
import ed_world_model.agents.patient as patient_module
import ed_world_model.agents.relative as relative_module
from ed_world_model.agents.nurse import NurseParserError, parse_nurse_response
from ed_world_model.agents.patient import (
    PatientAgent,
    PatientParserError,
    build_patient_prompt,
    parse_patient_response,
)
from ed_world_model.agents.relative import RelativeParserError, parse_relative_response
from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentProposal,
    VerbalOnlyProposal,
)


def _observation() -> dict:
    return {
        "patient_internal_state": {
            "chief_complaint": "shortness of breath",
            "symptoms": ["chest tightness", "dizziness"],
            "hidden_history": ["hypertension"],
            "hidden_allergies": ["penicillin"],
            "hidden_home_medications": ["lisinopril"],
            "disclosure_rules": "Volunteer breathing symptoms; reveal allergy if asked.",
        },
        "patient_emotion": {"label": "fear", "intensity": "high"},
        "recent_messages": [
            {
                "speaker": "clinician",
                "recipient": "patient",
                "content": "Can you tell me what you feel?",
            }
        ],
        "communication_ability": {"can_speak": True},
    }


def _prompt() -> str:
    return build_patient_prompt(
        _observation(),
        profile=AgentProfile(
            role="patient",
            name="Alex",
            traits={"communication_style": "brief", "anxiety": "high"},
        ),
        emotion_context={"label": "fear", "intensity": "high"},
    )


def _json_output(verbal_action=None) -> str:
    return json.dumps({"verbal_action": verbal_action})


def test_build_prompt_includes_patient_role() -> None:
    prompt = _prompt()

    assert "You are the patient in an ED simulation." in prompt
    assert '"role": "patient"' in prompt


def test_build_prompt_includes_patient_observation() -> None:
    prompt = _prompt()

    assert "Role-specific observation" in prompt
    assert "shortness of breath" in prompt
    assert "chest tightness" in prompt
    assert "Can you tell me what you feel?" in prompt


def test_build_prompt_does_not_include_full_global_state() -> None:
    prompt = _prompt()

    assert "GlobalState" not in prompt
    assert "truth_state" not in prompt
    assert "runtime_state" not in prompt


def test_build_prompt_includes_emotion_context_when_provided() -> None:
    prompt = _prompt()

    assert "Patient emotion context" in prompt
    assert "fear" in prompt
    assert "high" in prompt


def test_build_prompt_explains_hidden_fields_are_patient_owned_truth() -> None:
    prompt = _prompt()

    assert "hidden_* fields are patient-owned truth" in prompt
    assert "not hidden from you" in prompt
    assert "hidden from the clinical team" in prompt


def test_build_prompt_mentions_disclosure_rules_as_guidance() -> None:
    prompt = _prompt()

    assert "disclosure_rules are free-form guidance" in prompt
    assert "what to volunteer" in prompt
    assert "what to reveal only if asked" in prompt


def test_build_prompt_says_patient_may_stay_silent() -> None:
    assert "stay silent" in _prompt()


def test_build_prompt_says_no_internal_reasoning_or_chain_of_thought() -> None:
    prompt = _prompt()

    assert "silently reason" in prompt
    assert "profile traits" in prompt
    assert "verbal_action.content should reflect your AgentProfile traits" in prompt
    assert "Do not include internal reasoning." in prompt
    assert "Do not include chain-of-thought." in prompt
    assert "Do not output the silent reasoning" in prompt


def test_build_prompt_includes_prompt_level_anti_repetition_with_exceptions() -> None:
    prompt = _prompt()

    assert "Avoid repeating the same information" in prompt
    assert "same information, reassurance, question, concern, or instruction" in prompt
    assert "If you have nothing new or useful to add" in prompt
    assert "Repetition is allowed when you are directly asked again" in prompt
    assert "correcting a misunderstanding" in prompt
    assert "new clinical or conversation context makes repetition necessary" in prompt


def test_parse_valid_patient_verbal_action() -> None:
    proposal = parse_patient_response(
        _json_output(
            verbal_action={
                "speaker": "patient",
                "target": "clinician",
                "content": "My chest feels tight.",
                "requires_response": True,
            }
        )
    )

    assert isinstance(proposal, VerbalOnlyProposal)
    assert proposal.action is None
    assert proposal.verbal_action is not None
    assert proposal.verbal_action.recipient == "clinician"
    assert proposal.verbal_action.content == "My chest feels tight."


def test_parse_null_patient_verbal_action() -> None:
    proposal = parse_patient_response(_json_output(verbal_action=None))

    assert proposal.verbal_action is None
    assert proposal.action is None


def test_reject_action_field() -> None:
    with pytest.raises(PatientParserError, match="action"):
        parse_patient_response({"verbal_action": None, "action": None})


def test_reject_raw_text() -> None:
    with pytest.raises(PatientParserError, match="raw_text"):
        parse_patient_response(
            {
                "verbal_action": {
                    "speaker": "patient",
                    "target": "clinician",
                    "content": "I feel bad.",
                    "requires_response": False,
                    "raw_text": "legacy text",
                }
            }
        )


def test_reject_internal_reasoning_and_chain_of_thought() -> None:
    with pytest.raises(PatientParserError, match="internal_reasoning"):
        parse_patient_response(
            {"verbal_action": None, "internal_reasoning": "private thought"}
        )

    with pytest.raises(PatientParserError, match="chain_of_thought"):
        parse_patient_response(
            {
                "verbal_action": {
                    "speaker": "patient",
                    "target": "clinician",
                    "content": "I feel dizzy.",
                    "requires_response": False,
                    "chain_of_thought": "...",
                }
            }
        )


def test_reject_medical_diagnostic_and_behavior_actions() -> None:
    for forbidden_key in (
        "medical_treatment_order",
        "diagnostic_order",
        "behavior_action",
    ):
        with pytest.raises(PatientParserError, match=forbidden_key):
            parse_patient_response({"verbal_action": None, forbidden_key: {}})


def test_reject_invalid_json() -> None:
    with pytest.raises(PatientParserError, match="valid strict JSON"):
        parse_patient_response('{"verbal_action": null')


def test_reject_patient_speaker_mismatch() -> None:
    with pytest.raises(PatientParserError, match="speaker"):
        parse_patient_response(
            _json_output(
                {
                    "speaker": "clinician",
                    "target": "patient",
                    "content": "This is the wrong speaker.",
                    "requires_response": False,
                }
            )
        )


def test_reject_invalid_target_value() -> None:
    with pytest.raises(PatientParserError, match="target"):
        parse_patient_response(
            _json_output(
                {
                    "speaker": "patient",
                    "target": "doctor",
                    "content": "Who are you?",
                    "requires_response": True,
                }
            )
        )


def test_reject_target_and_recipient_both_present() -> None:
    with pytest.raises(PatientParserError, match="both target and recipient"):
        parse_patient_response(
            _json_output(
                {
                    "speaker": "patient",
                    "target": "clinician",
                    "recipient": "nurse",
                    "content": "I feel dizzy.",
                    "requires_response": False,
                }
            )
        )


def test_reject_extra_top_level_key() -> None:
    with pytest.raises(PatientParserError, match="only verbal_action"):
        parse_patient_response({"verbal_action": None, "confidence": 0.7})


def test_reject_extra_field_inside_verbal_action() -> None:
    with pytest.raises(PatientParserError, match="tone"):
        parse_patient_response(
            _json_output(
                {
                    "speaker": "patient",
                    "target": "clinician",
                    "content": "I feel dizzy.",
                    "requires_response": False,
                    "tone": "worried",
                }
            )
        )


def test_reject_duplicate_json_keys() -> None:
    with pytest.raises(PatientParserError, match="valid strict JSON"):
        parse_patient_response('{"verbal_action": null, "verbal_action": null}')


def test_reject_json_constants() -> None:
    with pytest.raises(PatientParserError, match="valid strict JSON"):
        parse_patient_response(
            '{"verbal_action": {"speaker": "patient", "target": null, '
            '"content": NaN, "requires_response": false}}'
        )


def test_fake_llm_callable_generate_path_returns_parsed_proposal() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return _json_output(
            verbal_action={
                "speaker": "patient",
                "target": "clinician",
                "content": "I am scared and short of breath.",
                "requires_response": True,
            }
        )

    agent = PatientAgent(
        fake_llm,
        profile=AgentProfile(role="patient", traits={"communication_style": "brief"}),
    )

    proposal = agent.generate(
        _observation(),
        emotion_context={"label": "fear", "intensity": "high"},
    )

    assert isinstance(proposal, AgentProposal)
    assert len(prompts) == 1
    assert "You are the patient in an ED simulation." in prompts[0]
    assert "Profile traits" in prompts[0]
    assert proposal.verbal_action.recipient == "clinician"
    assert proposal.action is None


def test_no_real_llm_api_calls_or_imports() -> None:
    source = inspect.getsource(patient_module)

    assert "openai" not in source.lower()
    assert "requests" not in source
    assert "urllib" not in source


def test_patient_nurse_relative_parsers_output_verbal_only_proposals() -> None:
    patient = parse_patient_response(
        _json_output(
            {
                "speaker": "patient",
                "target": "clinician",
                "content": "It hurts.",
                "requires_response": False,
            }
        )
    )
    nurse = parse_nurse_response(
        _json_output(
            {
                "speaker": "nurse",
                "target": "clinician",
                "content": "The ECG result is available.",
                "requires_response": False,
            }
        )
    )
    relative = parse_relative_response(
        _json_output(
            {
                "speaker": "relative",
                "target": "clinician",
                "content": "Is he going to be okay?",
                "requires_response": True,
            }
        )
    )

    for proposal in (patient, nurse, relative):
        assert isinstance(proposal, VerbalOnlyProposal)
        assert proposal.action is None


def test_patient_nurse_relative_speaker_mismatch_is_rejected_for_each_role() -> None:
    with pytest.raises(PatientParserError, match="speaker"):
        parse_patient_response(
            _json_output(
                {
                    "speaker": "clinician",
                    "target": "patient",
                    "content": "Wrong speaker.",
                    "requires_response": False,
                }
            )
        )
    with pytest.raises(NurseParserError, match="speaker"):
        parse_nurse_response(
            _json_output(
                {
                    "speaker": "patient",
                    "target": "clinician",
                    "content": "Wrong speaker.",
                    "requires_response": False,
                }
            )
        )
    with pytest.raises(RelativeParserError, match="speaker"):
        parse_relative_response(
            _json_output(
                {
                    "speaker": "nurse",
                    "target": "clinician",
                    "content": "Wrong speaker.",
                    "requires_response": False,
                }
            )
        )


def test_patient_nurse_relative_modules_do_not_import_transition_engines_or_global_state() -> None:
    source = "\n".join(
        [
            inspect.getsource(patient_module),
            inspect.getsource(nurse_module),
            inspect.getsource(relative_module),
        ]
    )

    assert "transition_engines" not in source
    assert "GlobalState" not in source
