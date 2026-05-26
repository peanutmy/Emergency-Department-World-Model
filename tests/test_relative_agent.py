from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.agents.relative as relative_module
from ed_world_model.agents.relative import (
    RelativeAgent,
    RelativeParserError,
    build_relative_prompt,
    parse_relative_response,
)
from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentProposal,
    VerbalOnlyProposal,
)


def _observation() -> dict:
    return {
        "relationship_profile": {"role": "relative", "name": "Sam"},
        "recent_messages": [
            {
                "speaker": "clinician",
                "recipient": "relative",
                "content": "Do you know what medications he takes?",
            }
        ],
        "visible_patient_status": {
            "status_flags": {"is_conscious": True},
            "visible_features": {"appearance": "pale", "work_of_breathing": "labored"},
        },
        "visible_last_turn_events": [],
        "family_side_hidden_info": None,
    }


def _prompt() -> str:
    return build_relative_prompt(
        _observation(),
        profile=AgentProfile(
            role="relative",
            name="Sam",
            traits={"relationship_role": "spouse", "communication_style": "anxious"},
        ),
    )


def _json_output(verbal_action=None) -> str:
    return json.dumps({"verbal_action": verbal_action})


def test_build_prompt_includes_relative_role() -> None:
    prompt = _prompt()

    assert "You are the patient's relative/family member in an ED simulation." in prompt
    assert '"role": "relative"' in prompt


def test_build_prompt_says_relative_is_verbal_only() -> None:
    assert "You are verbal-only in v1.3.1." in _prompt()


def test_build_prompt_does_not_include_full_global_state() -> None:
    prompt = _prompt()

    assert "GlobalState" not in prompt
    assert "truth_state" not in prompt
    assert "runtime_state" not in prompt


def test_build_prompt_says_no_behavior_actions_in_v1_3_1() -> None:
    prompt = _prompt()

    assert "do not perform behavior actions" in prompt.lower()
    assert "block care" in prompt
    assert "v1.3.1" in prompt


def test_build_prompt_says_relative_may_stay_silent() -> None:
    assert "stay silent" in _prompt()


def test_build_prompt_says_relative_should_not_invent_hidden_clinical_facts() -> None:
    prompt = _prompt()

    assert "Do not invent hidden clinical facts." in prompt
    assert "Do not read monitor-level exact vitals unless observation explicitly provides them." in prompt


def test_build_prompt_instructs_silent_reasoning_and_profile_traits() -> None:
    prompt = _prompt()

    assert "silently reason" in prompt
    assert "profile traits" in prompt
    assert "verbal_action.content should reflect your AgentProfile traits" in prompt
    assert "No relative emotion transition is modeled in v1.3.1." in prompt
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
    assert "ordinary answers, symptom statements" in prompt
    assert "Relative concerns should normally use requires_response=false" in prompt
    assert "unless you explicitly ask a question" in prompt


def test_parse_valid_relative_verbal_action() -> None:
    proposal = parse_relative_response(
        _json_output(
            verbal_action={
                "speaker": "relative",
                "target": "clinician",
                "content": "He takes a blood pressure pill at home.",
                "requires_response": False,
            }
        )
    )

    assert isinstance(proposal, VerbalOnlyProposal)
    assert proposal.action is None
    assert proposal.verbal_action is not None
    assert proposal.verbal_action.recipient == "clinician"


def test_relative_concern_requires_response_true_normalizes_false() -> None:
    proposal = parse_relative_response(
        _json_output(
            verbal_action={
                "speaker": "relative",
                "target": "clinician",
                "content": "I am really worried about him.",
                "requires_response": True,
            }
        )
    )

    assert proposal.verbal_action is not None
    assert proposal.verbal_action.requires_response is False


def test_relative_explicit_question_can_keep_requires_response_true() -> None:
    proposal = parse_relative_response(
        _json_output(
            verbal_action={
                "speaker": "relative",
                "target": "clinician",
                "content": "Is he going to be okay?",
                "requires_response": True,
            }
        )
    )

    assert proposal.verbal_action is not None
    assert proposal.verbal_action.requires_response is True


def test_parse_null_relative_verbal_action() -> None:
    proposal = parse_relative_response(_json_output(verbal_action=None))

    assert proposal.verbal_action is None
    assert proposal.action is None


def test_reject_action_field() -> None:
    with pytest.raises(RelativeParserError, match="action"):
        parse_relative_response({"verbal_action": None, "action": None})


def test_reject_raw_text() -> None:
    with pytest.raises(RelativeParserError, match="raw_text"):
        parse_relative_response(
            {
                "verbal_action": {
                    "speaker": "relative",
                    "target": "clinician",
                    "content": "Please help him.",
                    "requires_response": True,
                    "raw_text": "legacy text",
                }
            }
        )


def test_reject_internal_reasoning_and_chain_of_thought() -> None:
    with pytest.raises(RelativeParserError, match="internal_reasoning"):
        parse_relative_response(
            {"verbal_action": None, "internal_reasoning": "private thought"}
        )

    with pytest.raises(RelativeParserError, match="chain_of_thought"):
        parse_relative_response(
            {
                "verbal_action": {
                    "speaker": "relative",
                    "target": "clinician",
                    "content": "What is happening?",
                    "requires_response": True,
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
        with pytest.raises(RelativeParserError, match=forbidden_key):
            parse_relative_response({"verbal_action": None, forbidden_key: {}})


def test_reject_relative_speaker_mismatch() -> None:
    with pytest.raises(RelativeParserError, match="speaker"):
        parse_relative_response(
            _json_output(
                {
                    "speaker": "nurse",
                    "target": "clinician",
                    "content": "This is the wrong speaker.",
                    "requires_response": False,
                }
            )
        )


def test_reject_invalid_target_value() -> None:
    with pytest.raises(RelativeParserError, match="target"):
        parse_relative_response(
            _json_output(
                {
                    "speaker": "relative",
                    "target": "doctor",
                    "content": "What is happening?",
                    "requires_response": True,
                }
            )
        )


def test_reject_target_and_recipient_both_present() -> None:
    with pytest.raises(RelativeParserError, match="both target and recipient"):
        parse_relative_response(
            _json_output(
                {
                    "speaker": "relative",
                    "target": "clinician",
                    "recipient": "patient",
                    "content": "What is happening?",
                    "requires_response": True,
                }
            )
        )


def test_reject_extra_top_level_key() -> None:
    with pytest.raises(RelativeParserError, match="only verbal_action"):
        parse_relative_response({"verbal_action": None, "confidence": 0.7})


def test_reject_extra_field_inside_verbal_action() -> None:
    with pytest.raises(RelativeParserError, match="tone"):
        parse_relative_response(
            _json_output(
                {
                    "speaker": "relative",
                    "target": "clinician",
                    "content": "What is happening?",
                    "requires_response": True,
                    "tone": "worried",
                }
            )
        )


def test_reject_duplicate_json_keys() -> None:
    with pytest.raises(RelativeParserError, match="valid strict JSON"):
        parse_relative_response('{"verbal_action": null, "verbal_action": null}')


def test_reject_json_constants() -> None:
    with pytest.raises(RelativeParserError, match="valid strict JSON"):
        parse_relative_response(
            '{"verbal_action": {"speaker": "relative", "target": null, '
            '"content": NaN, "requires_response": false}}'
        )


def test_fake_llm_callable_generate_path_returns_parsed_proposal() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return _json_output(
            verbal_action={
                "speaker": "relative",
                "target": "clinician",
                "content": "I am worried. What happens next?",
                "requires_response": True,
            }
        )

    agent = RelativeAgent(
        fake_llm,
        profile=AgentProfile(role="relative", traits={"relationship_role": "spouse"}),
    )

    proposal = agent.generate(_observation())

    assert isinstance(proposal, AgentProposal)
    assert len(prompts) == 1
    assert "You are the patient's relative/family member in an ED simulation." in prompts[0]
    assert "Profile traits" in prompts[0]
    assert proposal.verbal_action.recipient == "clinician"
    assert proposal.action is None


def test_no_real_llm_api_calls_or_imports() -> None:
    source = inspect.getsource(relative_module)

    assert "openai" not in source.lower()
    assert "requests" not in source
    assert "urllib" not in source
