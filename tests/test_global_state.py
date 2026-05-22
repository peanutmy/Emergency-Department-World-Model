from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.state.global_state import (
    Demographics,
    DiagnosticResult,
    Event,
    GlobalState,
    KnownFacts,
    Message,
    PatientEmotion,
    RuntimeState,
    TestBankItem as StateTestBankItem,
)


def test_can_construct_minimal_global_state() -> None:
    state = GlobalState()

    assert state.truth_state.scenario_description is None
    assert state.truth_state.demographics == Demographics()
    assert state.patient_state.status_flags.is_alive is True
    assert state.known_facts == KnownFacts()
    assert state.runtime_state.turn_index == 0


def test_demographics_defaults_are_none() -> None:
    state = GlobalState()

    assert state.truth_state.demographics.name is None
    assert state.truth_state.demographics.age is None
    assert state.truth_state.demographics.sex is None
    assert state.truth_state.demographics.weight_kg is None


def test_demographics_store_stable_patient_truth() -> None:
    state = GlobalState(
        truth_state={
            "demographics": {
                "name": "Alex Morgan",
                "age": 67,
                "sex": "female",
                "weight_kg": 72.5,
            }
        }
    )

    assert state.truth_state.demographics == Demographics(
        name="Alex Morgan",
        age=67,
        sex="female",
        weight_kg=72.5,
    )


def test_defaults_are_safe_and_empty_where_appropriate() -> None:
    first = GlobalState()
    second = GlobalState()

    first.known_facts.known_history.append("hypertension")
    first.runtime_state.messages.append(Message(speaker="patient", content="Hello."))

    assert second.known_facts.known_history == []
    assert second.runtime_state.messages == []
    assert second.truth_state.test_bank == []
    assert second.runtime_state.pending_questions == []
    assert second.runtime_state.pending_diagnostic_results == []
    assert second.runtime_state.newly_available_results == []
    assert second.runtime_state.last_turn_events == []
    assert second.runtime_state.current_turn_events == []


def test_test_bank_item_supports_turnaround_turns() -> None:
    item = StateTestBankItem(name="ECG", result="LVH and A.fib", turnaround_turns=1)
    default_turnaround = StateTestBankItem(name="VBG", result="pH 7.30")

    assert item.name == "ECG"
    assert item.result == "LVH and A.fib"
    assert item.turnaround_turns == 1
    assert default_turnaround.turnaround_turns is None


def test_runtime_state_separates_messages_from_last_turn_events() -> None:
    state = GlobalState(
        runtime_state=RuntimeState(
            messages=[
                Message(
                    speaker="clinician",
                    recipient="patient",
                    content="How are you feeling?",
                    turn_index=0,
                )
            ],
            last_turn_events=[
                Event(
                    type="diagnostic_release",
                    turn_index=0,
                    payload={"name": "ECG"},
                )
            ],
            current_turn_events=[
                Event(
                    type="validation_drop",
                    turn_index=1,
                    payload={"reason": "invalid"},
                )
            ],
        )
    )

    assert state.runtime_state.messages[0].content == "How are you feeling?"
    assert state.runtime_state.last_turn_events[0].type == "diagnostic_release"
    assert state.runtime_state.current_turn_events[0].type == "validation_drop"
    assert len(state.runtime_state.messages) == 1
    assert len(state.runtime_state.last_turn_events) == 1
    assert len(state.runtime_state.current_turn_events) == 1


def test_known_facts_is_the_clinical_team_fact_store() -> None:
    state = GlobalState(
        known_facts=KnownFacts(
            known_history=["hypertension"],
            known_allergies=["penicillin"],
            known_medications=["metformin"],
            known_symptoms=["shortness of breath"],
            available_results=[
                DiagnosticResult(name="ECG", result="LVH and A.fib")
            ],
        )
    )

    assert "known_facts" in GlobalState.model_fields
    assert "memory" not in GlobalState.model_fields
    assert "case_memory" not in GlobalState.model_fields
    assert "clinical_facts" not in GlobalState.model_fields
    assert "memory" not in RuntimeState.model_fields
    assert state.known_facts.available_results[0].name == "ECG"


def test_json_model_serialization_round_trip() -> None:
    state = GlobalState(
        truth_state={
            "scenario_description": "Shortness of breath case",
            "demographics": {
                "name": "Alex Morgan",
                "age": 67,
                "sex": "female",
                "weight_kg": 72.5,
            },
            "test_bank": [
                {
                    "name": "Chest X-ray",
                    "result": "CHF",
                    "turnaround_turns": 2,
                }
            ],
        },
        known_facts={
            "available_results": [
                {
                    "name": "ECG",
                    "result": "LVH and A.fib",
                }
            ]
        },
        runtime_state={
            "messages": [
                {
                    "speaker": "nurse",
                    "recipient": "patient",
                    "content": "I am placing oxygen now.",
                }
            ],
            "last_turn_events": [
                {
                    "type": "nurse_shadow_execution",
                    "payload": {"ordered_by": "clinician", "executed_by": "nurse"},
                }
            ],
        },
    )

    serialized = state.model_dump_json()
    loaded = GlobalState.model_validate_json(serialized)

    assert loaded == state
    assert loaded.model_dump()["truth_state"]["demographics"] == {
        "name": "Alex Morgan",
        "age": 67,
        "sex": "female",
        "weight_kg": 72.5,
    }
    assert loaded.model_dump()["truth_state"]["test_bank"][0]["name"] == "Chest X-ray"
    assert loaded.runtime_state.messages[0].speaker == "nurse"


def test_raw_text_is_not_required_for_global_state_construction() -> None:
    state = GlobalState()

    assert "raw_text" not in GlobalState.model_fields
    assert state.runtime_state.messages == []


def test_patient_internal_state_is_static_hidden_dialogue_truth() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "chief_complaint": "shortness of breath",
                "symptoms": ["shortness of breath", "chest discomfort"],
                "hidden_history": ["hypertension"],
                "hidden_allergies": ["penicillin"],
                "hidden_home_medications": ["metformin"],
                "disclosure_rules": (
                    "Volunteer dyspnea. Mention chest discomfort only if asked."
                ),
            }
        }
    )

    internal_state = state.truth_state.patient_internal_state

    assert internal_state.symptoms == ["shortness of breath", "chest discomfort"]
    assert internal_state.disclosure_rules is not None
    assert "patient_internal_state" not in type(state.patient_state).model_fields
    assert "patient_internal_state" not in type(state.psych_state).model_fields


def test_patient_emotion_intensity_is_categorical() -> None:
    assert PatientEmotion(intensity="low").intensity == "low"
    assert PatientEmotion(intensity="medium").intensity == "medium"
    assert PatientEmotion(intensity="high").intensity == "high"
    assert PatientEmotion(intensity=None).intensity is None

    with pytest.raises(ValueError):
        PatientEmotion(intensity=0.5)


def test_event_payload_is_not_durable_state_store() -> None:
    state = GlobalState(
        runtime_state={
            "last_turn_events": [
                {
                    "type": "validation_drop",
                    "payload": {
                        "reason": "invalid diagnostic order",
                        "raw_text": "legacy metadata is not runtime state",
                    },
                }
            ]
        }
    )

    assert state.runtime_state.last_turn_events[0].payload == {
        "reason": "invalid diagnostic order",
        "raw_text": "legacy metadata is not runtime state",
    }
    assert state.known_facts == KnownFacts()
    assert "payload" not in GlobalState.model_fields
