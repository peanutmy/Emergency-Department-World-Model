from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.agents.schemas import (
    AgentProfile,
    AgentProposal,
    AgentRuntimeInput,
    VerbalAction,
    VerbalOnlyProposal,
)


def test_agent_runtime_input_does_not_require_global_state() -> None:
    runtime_input = AgentRuntimeInput(
        role="clinician",
        observation={"known_facts": {"known_symptoms": ["dyspnea"]}},
    )

    assert runtime_input.observation["known_facts"]["known_symptoms"] == ["dyspnea"]
    assert "GlobalState" not in AgentRuntimeInput.model_fields
    assert "global_state" not in AgentRuntimeInput.model_fields


def test_agent_runtime_input_accepts_partial_observation_dict() -> None:
    runtime_input = AgentRuntimeInput(
        role="patient",
        observation={
            "patient_internal_state": {"chief_complaint": "shortness of breath"},
            "communication_ability": {"can_speak": True},
        },
        turn_index=3,
    )

    assert runtime_input.role == "patient"
    assert runtime_input.observation["communication_ability"]["can_speak"] is True
    assert runtime_input.turn_index == 3


def test_agent_profile_supports_only_role_name_and_traits_fields() -> None:
    profile = AgentProfile(
        role="patient",
        name="Alex",
        traits={
            "communication_style": "brief answers",
            "notes": "stored inside traits when needed",
        },
    )

    assert set(AgentProfile.model_fields) == {"role", "name", "traits"}
    assert profile.traits["communication_style"] == "brief answers"
    assert profile.traits["notes"] == "stored inside traits when needed"


def test_agent_profile_rejects_top_level_communication_style_or_notes() -> None:
    with pytest.raises(ValueError):
        AgentProfile(role="patient", communication_style="brief")

    with pytest.raises(ValueError):
        AgentProfile(role="patient", notes="do not add top-level notes")


def test_agent_profile_traits_match_global_state_bounded_value_types() -> None:
    profile = AgentProfile(
        role="relative",
        traits={
            "relationship_role": "spouse",
            "years_in_relationship": 32,
            "stress_level_baseline": 0.4,
            "is_primary_contact": True,
            "languages": ["English", "Spanish"],
            "health_literacy": None,
        },
    )

    assert profile.traits == {
        "relationship_role": "spouse",
        "years_in_relationship": 32,
        "stress_level_baseline": 0.4,
        "is_primary_contact": True,
        "languages": ["English", "Spanish"],
        "health_literacy": None,
    }


def test_agent_profile_traits_reject_complex_nested_objects() -> None:
    with pytest.raises(ValueError):
        AgentProfile(
            role="patient",
            traits={"nested": {"communication_style": "brief"}},
        )


def test_verbal_action_serializes_and_deserializes() -> None:
    verbal_action = VerbalAction(
        speaker="clinician",
        recipient="patient",
        content="How are you feeling?",
        requires_response=True,
    )

    loaded = VerbalAction.model_validate_json(verbal_action.model_dump_json())

    assert loaded == verbal_action
    assert loaded.message_recipient == "patient"


def test_agent_proposal_can_represent_verbal_only_output() -> None:
    proposal = AgentProposal(
        verbal_action=VerbalAction(speaker="patient", content="I feel short of breath.")
    )

    assert proposal.verbal_action is not None
    assert proposal.action is None


def test_agent_proposal_can_represent_clinician_verbal_and_action_output() -> None:
    proposal = AgentProposal(
        verbal_action=VerbalAction(
            speaker="clinician",
            recipient="nurse",
            content="Please place oxygen.",
        ),
        action={
            "type": "medical_treatment_order",
            "family": "respiratory_support",
            "kind_hint": "oxygen_support",
            "params": {"oxygen_device": "NRB", "FiO2": None},
        },
    )

    assert proposal.verbal_action is not None
    assert proposal.action["type"] == "medical_treatment_order"
    assert proposal.action["params"]["FiO2"] is None


def test_verbal_only_proposal_rejects_action() -> None:
    with pytest.raises(ValueError):
        VerbalOnlyProposal(
            verbal_action=VerbalAction(speaker="nurse", content="I am here."),
            action={"type": "medical_treatment_order"},
        )
