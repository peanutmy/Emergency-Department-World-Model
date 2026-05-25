from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.agents.prompting as prompting_module
import ed_world_model.agents.schemas as schemas_module
from ed_world_model.agents.prompting import (
    build_agent_prompt,
    build_clinician_prompt,
    build_nurse_prompt,
    build_patient_prompt,
    build_relative_prompt,
    parse_agent_proposal,
)
from ed_world_model.agents.schemas import AgentProfile, AgentRuntimeInput


def test_prompt_includes_profile_emotion_observation_recent_messages_and_common_rules() -> None:
    runtime_input = AgentRuntimeInput(
        role="clinician",
        observation={"known_facts": {"known_symptoms": ["dyspnea"]}},
        profile=AgentProfile(
            role="clinician",
            name="Dr. Lee",
            traits={"experience_level": "attending", "communication_style": "direct"},
        ),
        emotion_context={"label": "worried", "intensity": "medium"},
        recent_messages=[
            {
                "speaker": "patient",
                "recipient": "clinician",
                "content": "I cannot breathe.",
            }
        ],
        turn_index=4,
    )

    prompt = build_agent_prompt(runtime_input)

    assert "Role: clinician" in prompt
    assert "Profile traits" in prompt
    assert "experience_level" in prompt
    assert "communication_style" in prompt
    assert "Emotion context" in prompt
    assert "worried" in prompt
    assert "known_symptoms" in prompt
    assert "I cannot breathe." in prompt
    assert (
        "The following observation is a partial role-specific view of the world. "
        "Do not assume access to hidden state not shown here."
    ) in prompt
    assert "may stay silent" in prompt
    assert "Avoid repeating the same information" in prompt
    assert "Repetition is allowed when you are directly asked again" in prompt
    assert "correcting a misunderstanding" in prompt
    assert "new clinical or conversation context makes repetition necessary" in prompt
    assert "Output final structured JSON only." in prompt
    assert "Do not include internal reasoning in the output or store it." in prompt
    assert "Do not include chain-of-thought in the output or store it." in prompt
    assert "Do not include raw_text in the output." in prompt


def test_clinician_prompt_mentions_action_choices_and_runtime_action_rules() -> None:
    prompt = build_clinician_prompt(
        AgentRuntimeInput(
            role="clinician",
            observation={"available_diagnostic_tests": ["ECG"]},
        )
    )

    assert "medical_treatment_order" in prompt
    assert "diagnostic_order" in prompt
    assert "null" in prompt
    assert "no_action is system-generated and is not selectable" in prompt
    assert "raw_text" in prompt
    assert "family -> kind_hint -> params" in prompt
    assert "one test_name only" in prompt
    assert "params may be null if unknown or unspecified" in prompt.lower()


def test_patient_prompt_includes_emotion_and_patient_internal_state_guidance() -> None:
    prompt = build_patient_prompt(
        AgentRuntimeInput(
            role="patient",
            observation={
                "patient_internal_state": {
                    "chief_complaint": "shortness of breath",
                    "hidden_history": ["hypertension"],
                    "disclosure_rules": "Mention chest pain only if asked.",
                },
                "communication_ability": {"can_speak": True},
            },
            emotion_context={"label": "fear", "intensity": "high"},
        )
    )

    assert "Patient emotion context" in prompt
    assert "fear" in prompt
    assert "patient_internal_state" in prompt
    assert "shortness of breath" in prompt
    assert "disclosure_rules are free-form guidance" in prompt
    assert "not deterministic rules" in prompt
    assert "Do not reveal hidden facts unless the observation and disclosure_rules allow it." in prompt
    assert "be unable to answer, or stay silent" in prompt
    assert "GlobalState" not in prompt
    assert "truth_state" not in prompt


def test_nurse_prompt_is_verbal_only_and_no_physical_action() -> None:
    prompt = build_nurse_prompt(
        AgentRuntimeInput(
            role="nurse",
            observation={"newly_available_results": [{"name": "ECG"}]},
        )
    )

    assert "verbal-only" in prompt
    assert "no physical action" in prompt
    assert "must not return an action" in prompt
    assert "result-related info" in prompt
    assert "bedside reassurance" in prompt
    assert "Stay silent" in prompt


def test_relative_prompt_has_no_behavior_action_or_emotion_transition() -> None:
    prompt = build_relative_prompt(
        AgentRuntimeInput(
            role="relative",
            observation={"relationship_profile": {"role": "relative"}},
        )
    )

    assert "verbal-only" in prompt
    assert "no behavior action" in prompt
    assert "Stay silent unless asked or explicitly selected" in prompt
    assert "no relative emotion transition in v1.3.1" in prompt


def test_parser_parses_json_into_agent_proposal_and_preserves_nulls() -> None:
    proposal = parse_agent_proposal(
        json.dumps(
            {
                "verbal_action": None,
                "action": {
                    "type": "medical_treatment_order",
                    "family": "respiratory_support",
                    "kind_hint": "future_kind_hint_validator_handles_this",
                    "params": {"dose": None},
                },
            }
        ),
        role="clinician",
    )

    assert proposal.verbal_action is None
    assert proposal.action["params"]["dose"] is None
    assert proposal.action["kind_hint"] == "future_kind_hint_validator_handles_this"


def test_parser_rejects_raw_text_anywhere() -> None:
    with pytest.raises(ValueError, match="raw_text"):
        parse_agent_proposal(
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "content": "I will help.",
                    "metadata": {"raw_text": "legacy text"},
                }
            },
            role="clinician",
        )


def test_parser_rejects_internal_reasoning_and_chain_of_thought_fields() -> None:
    with pytest.raises(ValueError, match="internal_reasoning"):
        parse_agent_proposal(
            {
                "internal_reasoning": "hidden deliberation",
                "verbal_action": None,
                "action": None,
            },
            role="clinician",
        )

    with pytest.raises(ValueError, match="chain_of_thought"):
        parse_agent_proposal(
            {
                "verbal_action": None,
                "action": {"type": "diagnostic_order", "chain_of_thought": "..."},
            },
            role="clinician",
        )


def test_parser_rejects_action_for_verbal_only_roles() -> None:
    with pytest.raises(ValueError, match="verbal-only"):
        parse_agent_proposal(
            {
                "verbal_action": {"speaker": "nurse", "content": "I am here."},
                "action": {"type": "medical_treatment_order"},
            },
            role="nurse",
        )


def test_parser_allows_null_action_for_verbal_only_roles() -> None:
    proposal = parse_agent_proposal(
        {
            "verbal_action": {"speaker": "relative", "content": "What is happening?"},
            "action": None,
        },
        role="relative",
    )

    assert proposal.action is None
    assert proposal.verbal_action.content == "What is happening?"


def test_prompting_and_schema_modules_do_not_import_llm_apis_or_transition_engines() -> None:
    source = "\n".join(
        [
            inspect.getsource(prompting_module),
            inspect.getsource(schemas_module),
        ]
    )

    assert "transition_engines" not in source
    assert "openai" not in source.lower()
    assert "requests" not in source
