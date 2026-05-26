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
from ed_world_model.agents.nurse import (
    NurseAgent,
    NurseParserError,
    build_nurse_prompt,
    parse_nurse_response,
)
from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentProposal,
    VerbalOnlyProposal,
)


def _observation() -> dict:
    return {
        "patient_state": {
            "vitals": {"HR": 118, "O2Sat": 91},
            "features": {"work_of_breathing": "increased"},
        },
        "recent_messages": [
            {
                "speaker": "clinician",
                "recipient": "nurse",
                "content": "Any new results?",
            }
        ],
        "newly_available_results": [{"name": "ECG", "result": "atrial fibrillation"}],
        "last_turn_events": [],
        "last_bedside_event_if_any": {"type": "nurse_shadow_execution"},
    }


def _prompt() -> str:
    return build_nurse_prompt(
        _observation(),
        profile=AgentProfile(
            role="nurse",
            name="Nurse Patel",
            traits={"communication_style": "calm", "experience_level": "senior"},
        ),
    )


def _json_output(verbal_action=None) -> str:
    return json.dumps({"verbal_action": verbal_action})


def test_build_prompt_includes_nurse_role() -> None:
    prompt = _prompt()

    assert "You are the bedside nurse in an ED simulation." in prompt
    assert '"role": "nurse"' in prompt


def test_build_prompt_says_nurse_is_verbal_only() -> None:
    assert "You are verbal-only in v1.3.1." in _prompt()


def test_build_prompt_does_not_include_full_global_state() -> None:
    prompt = _prompt()

    assert "GlobalState" not in prompt
    assert "truth_state" not in prompt
    assert "runtime_state" not in prompt


def test_build_prompt_says_nurse_has_no_physical_action() -> None:
    prompt = _prompt()

    assert "no physical action" in prompt
    assert "must not return an action" in prompt
    assert "do not execute physical actions directly" in prompt.lower()


def test_build_prompt_says_nurse_may_report_newly_available_results() -> None:
    prompt = _prompt()

    assert "newly available results" in prompt
    assert "ECG" in prompt
    assert "atrial fibrillation" in prompt


def test_build_prompt_says_nurse_may_stay_silent() -> None:
    assert "stay silent" in _prompt()


def test_build_prompt_instructs_silent_reasoning_and_profile_traits() -> None:
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


def test_build_prompt_restricts_requires_response_to_explicit_questions() -> None:
    prompt = _prompt()

    assert "requires_response=true only for an explicit question" in prompt
    assert "status reports, reassurance, bedside instructions" in prompt
    assert "Nurse reports to the clinician should normally use" in prompt
    assert "unless you explicitly ask a question" in prompt


def test_parse_valid_nurse_verbal_action() -> None:
    proposal = parse_nurse_response(
        _json_output(
            verbal_action={
                "speaker": "nurse",
                "target": "clinician",
                "content": "The ECG result is available.",
                "requires_response": False,
            }
        )
    )

    assert isinstance(proposal, VerbalOnlyProposal)
    assert proposal.action is None
    assert proposal.verbal_action is not None
    assert proposal.verbal_action.recipient == "clinician"


def test_nurse_status_report_requires_response_true_normalizes_false() -> None:
    proposal = parse_nurse_response(
        _json_output(
            verbal_action={
                "speaker": "nurse",
                "target": "clinician",
                "content": "The ECG result is available.",
                "requires_response": True,
            }
        )
    )

    assert proposal.verbal_action is not None
    assert proposal.verbal_action.requires_response is False


def test_nurse_explicit_question_can_keep_requires_response_true() -> None:
    proposal = parse_nurse_response(
        _json_output(
            verbal_action={
                "speaker": "nurse",
                "target": "clinician",
                "content": "Do you want me to repeat the blood pressure?",
                "requires_response": True,
            }
        )
    )

    assert proposal.verbal_action is not None
    assert proposal.verbal_action.requires_response is True


def test_parse_null_nurse_verbal_action() -> None:
    proposal = parse_nurse_response(_json_output(verbal_action=None))

    assert proposal.verbal_action is None
    assert proposal.action is None


def test_reject_action_field() -> None:
    with pytest.raises(NurseParserError, match="action"):
        parse_nurse_response({"verbal_action": None, "action": None})


def test_reject_raw_text() -> None:
    with pytest.raises(NurseParserError, match="raw_text"):
        parse_nurse_response(
            {
                "verbal_action": {
                    "speaker": "nurse",
                    "target": "patient",
                    "content": "I am right here.",
                    "requires_response": False,
                },
                "metadata": {"raw_text": "legacy text"},
            }
        )


def test_reject_internal_reasoning_and_chain_of_thought() -> None:
    with pytest.raises(NurseParserError, match="internal_reasoning"):
        parse_nurse_response(
            {"verbal_action": None, "internal_reasoning": "private thought"}
        )

    with pytest.raises(NurseParserError, match="chain_of_thought"):
        parse_nurse_response(
            {
                "verbal_action": {
                    "speaker": "nurse",
                    "target": "clinician",
                    "content": "I will update you.",
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
        with pytest.raises(NurseParserError, match=forbidden_key):
            parse_nurse_response({"verbal_action": None, forbidden_key: {}})


def test_reject_nurse_speaker_mismatch() -> None:
    with pytest.raises(NurseParserError, match="speaker"):
        parse_nurse_response(
            _json_output(
                {
                    "speaker": "patient",
                    "target": "clinician",
                    "content": "This is the wrong speaker.",
                    "requires_response": False,
                }
            )
        )


def test_reject_invalid_target_value() -> None:
    with pytest.raises(NurseParserError, match="target"):
        parse_nurse_response(
            _json_output(
                {
                    "speaker": "nurse",
                    "target": "doctor",
                    "content": "The result is back.",
                    "requires_response": False,
                }
            )
        )


def test_reject_target_and_recipient_both_present() -> None:
    with pytest.raises(NurseParserError, match="both target and recipient"):
        parse_nurse_response(
            _json_output(
                {
                    "speaker": "nurse",
                    "target": "clinician",
                    "recipient": "patient",
                    "content": "The result is back.",
                    "requires_response": False,
                }
            )
        )


def test_reject_extra_top_level_key() -> None:
    with pytest.raises(NurseParserError, match="only verbal_action"):
        parse_nurse_response({"verbal_action": None, "confidence": 0.7})


def test_reject_extra_field_inside_verbal_action() -> None:
    with pytest.raises(NurseParserError, match="tone"):
        parse_nurse_response(
            _json_output(
                {
                    "speaker": "nurse",
                    "target": "clinician",
                    "content": "The result is back.",
                    "requires_response": False,
                    "tone": "calm",
                }
            )
        )


def test_reject_duplicate_json_keys() -> None:
    with pytest.raises(NurseParserError, match="valid strict JSON"):
        parse_nurse_response('{"verbal_action": null, "verbal_action": null}')


def test_reject_json_constants() -> None:
    with pytest.raises(NurseParserError, match="valid strict JSON"):
        parse_nurse_response(
            '{"verbal_action": {"speaker": "nurse", "target": null, '
            '"content": NaN, "requires_response": false}}'
        )


def test_fake_llm_callable_generate_path_returns_parsed_proposal() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return _json_output(
            verbal_action={
                "speaker": "nurse",
                "target": "patient",
                "content": "Try to take slow breaths while we help.",
                "requires_response": False,
            }
        )

    agent = NurseAgent(
        fake_llm,
        profile=AgentProfile(role="nurse", traits={"bedside_manner": "steady"}),
    )

    proposal = agent.generate(_observation())

    assert isinstance(proposal, AgentProposal)
    assert len(prompts) == 1
    assert "You are the bedside nurse in an ED simulation." in prompts[0]
    assert "Profile traits" in prompts[0]
    assert proposal.verbal_action.recipient == "patient"
    assert proposal.action is None


def test_no_real_llm_api_calls_or_imports() -> None:
    source = inspect.getsource(nurse_module)

    assert "openai" not in source.lower()
    assert "requests" not in source
    assert "urllib" not in source
