from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.orchestration.observation_builder as observation_builder_module
from ed_world_model.orchestration.observation_builder import ObservationBuilder
from ed_world_model.state.global_state import DiagnosticResult, Event, GlobalState


UNRELEASED_ECG_RESULT = "SECRET UNRELEASED ECG RESULT"
UNRELEASED_TROPONIN_RESULT = "SECRET UNRELEASED TROPONIN RESULT"
HIDDEN_HISTORY = "secret diabetes history"
DISCLOSED_HISTORY = "disclosed asthma history"


def _rich_state() -> GlobalState:
    return GlobalState(
        truth_state={
            "scenario_description": "Adult patient with respiratory distress.",
            "demographics": {
                "name": "Alex Morgan",
                "age": 67,
                "sex": "female",
                "weight_kg": 72.5,
            },
            "patient_internal_state": {
                "chief_complaint": "shortness of breath",
                "symptoms": ["dyspnea", "chest tightness"],
                "hidden_history": [HIDDEN_HISTORY],
                "hidden_allergies": ["secret allergy"],
                "hidden_home_medications": ["secret medication"],
                "disclosure_rules": "Answer directly when asked about symptoms.",
            },
            "test_bank": [
                {
                    "name": "ECG",
                    "result": UNRELEASED_ECG_RESULT,
                    "turnaround_turns": 1,
                },
                {
                    "name": "Troponin",
                    "result": UNRELEASED_TROPONIN_RESULT,
                },
            ],
        },
        patient_state={
            "vitals": {
                "HR": 118,
                "BP_sys": 154,
                "BP_dia": 92,
                "RR": 28,
                "O2Sat": 89,
                "T": 37.2,
            },
            "features": {"work_of_breathing": "increased"},
            "status_flags": {
                "is_alive": True,
                "can_speak": True,
                "is_conscious": True,
            },
        },
        known_facts={
            "known_history": [
                {
                    "item": DISCLOSED_HISTORY,
                    "status": "present",
                    "source_texts": ["Family says she has asthma."],
                }
            ],
            "known_symptoms": [
                {
                    "name": "dyspnea",
                    "status": "present",
                    "source_texts": ["I feel short of breath."],
                }
            ],
            "available_results": [
                {"name": "Chest X-ray", "result": "released pulmonary edema"}
            ],
        },
        psych_state={
            "patient_emotion": {
                "label": "worried",
                "intensity": "medium",
            }
        },
        runtime_state={
            "messages": [
                {
                    "speaker": "clinician",
                    "recipient": "patient",
                    "content": "How are you feeling?",
                    "turn_index": 0,
                },
                {
                    "speaker": "relative",
                    "recipient": None,
                    "content": "She looks worse than usual.",
                    "turn_index": 0,
                },
                {
                    "speaker": "nurse",
                    "recipient": "clinician",
                    "content": "The monitor is cycling.",
                    "turn_index": 0,
                },
            ],
            "pending_questions": [
                {
                    "source_agent": "clinician",
                    "target_agent": "patient",
                    "question_text": "Do you have chest pain?",
                    "created_at_turn": 0,
                }
            ],
            "newly_available_results": [
                {"name": "VBG", "result": "released pH 7.30"}
            ],
            "last_turn_events": [
                {
                    "type": "diagnostic_release",
                    "turn_index": 0,
                    "payload": {"test_name": "VBG", "released_at_turn": 0},
                },
                {
                    "type": "nurse_shadow_execution",
                    "turn_index": 0,
                    "payload": {
                        "execution_mode": "shadow_execution",
                        "visible_to": ["patient", "relative"],
                    },
                },
            ],
        },
    )


def _observation_text(observation: object) -> str:
    return json.dumps(observation, sort_keys=True)


def test_build_for_returns_observations_only_for_active_agents() -> None:
    observations = ObservationBuilder().build_for(["patient", "relative"], _rich_state())

    assert set(observations) == {"patient", "relative"}
    assert "clinician" not in observations
    assert "nurse" not in observations


def test_clinician_observation_includes_patient_state() -> None:
    observation = ObservationBuilder().build_clinician_observation(_rich_state())

    assert observation["patient_state"]["vitals"]["HR"] == 118.0
    assert observation["patient_state"]["status_flags"]["can_speak"] is True


def test_clinician_observation_includes_known_facts() -> None:
    observation = ObservationBuilder().build_clinician_observation(_rich_state())

    assert observation["known_facts"]["known_history"] == [
        {
            "item": DISCLOSED_HISTORY,
            "status": "present",
            "source_texts": ["Family says she has asthma."],
        }
    ]
    assert observation["known_facts"]["known_symptoms"] == [
        {
            "name": "dyspnea",
            "status": "present",
            "onset": None,
            "severity": None,
            "source_texts": ["I feel short of breath."],
        }
    ]


def test_clinician_observation_includes_released_chief_complaint() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "chief_complaint": "hidden shortness of breath"
            }
        },
        known_facts={"chief_complaint": "shortness of breath"},
    )

    observation = ObservationBuilder().build_clinician_observation(state)

    assert observation["known_facts"]["chief_complaint"] == "shortness of breath"


def test_clinician_observation_does_not_leak_unreleased_chief_complaint() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "chief_complaint": "private patient-side chief complaint"
            }
        }
    )

    observation = ObservationBuilder().build_clinician_observation(state)

    assert "private patient-side chief complaint" not in _observation_text(observation)
    assert observation["known_facts"]["chief_complaint"] is None


def test_clinician_observation_does_not_include_raw_scenario_description() -> None:
    state = GlobalState(
        truth_state={
            "scenario_description": "DO NOT SHOW RAW SCENARIO DESCRIPTION",
        }
    )

    observation = ObservationBuilder().build_clinician_observation(state)

    assert "safe_case_context" not in observation
    assert "scenario_description" not in observation
    assert "DO NOT SHOW RAW SCENARIO DESCRIPTION" not in _observation_text(observation)


def test_clinician_observation_includes_available_diagnostic_test_names() -> None:
    observation = ObservationBuilder().build_clinician_observation(_rich_state())

    assert observation["available_diagnostic_tests"] == ["ECG", "Troponin"]


def test_clinician_observation_does_not_include_unreleased_test_results() -> None:
    observation = ObservationBuilder().build_clinician_observation(_rich_state())

    text = _observation_text(observation)
    assert UNRELEASED_ECG_RESULT not in text
    assert UNRELEASED_TROPONIN_RESULT not in text


def test_clinician_observation_hides_patient_internal_hidden_history() -> None:
    observation = ObservationBuilder().build_clinician_observation(_rich_state())

    text = _observation_text(observation)
    assert DISCLOSED_HISTORY in text
    assert HIDDEN_HISTORY not in text
    assert "patient_internal_state" not in observation


def test_nurse_observation_does_not_include_patient_internal_hidden_history() -> None:
    observation = ObservationBuilder().build_nurse_observation(_rich_state())

    assert HIDDEN_HISTORY not in _observation_text(observation)
    assert "patient_internal_state" not in observation


def test_nurse_observation_includes_newly_available_results() -> None:
    observation = ObservationBuilder().build_nurse_observation(_rich_state())

    assert observation["newly_available_results"] == [
        {"name": "VBG", "result": "released pH 7.30"}
    ]


def test_nurse_observation_does_not_include_unreleased_test_results() -> None:
    observation = ObservationBuilder().build_nurse_observation(_rich_state())

    text = _observation_text(observation)
    assert UNRELEASED_ECG_RESULT not in text
    assert UNRELEASED_TROPONIN_RESULT not in text


def test_patient_observation_includes_patient_internal_subset() -> None:
    observation = ObservationBuilder().build_patient_observation(_rich_state())

    assert observation["patient_internal_state"] == {
        "chief_complaint": "shortness of breath",
        "symptoms": ["dyspnea", "chest tightness"],
        "hidden_history": [HIDDEN_HISTORY],
        "hidden_allergies": ["secret allergy"],
        "hidden_home_medications": ["secret medication"],
        "disclosure_rules": "Answer directly when asked about symptoms.",
    }


def test_patient_observation_includes_patient_owned_hidden_facts() -> None:
    patient_internal_state = ObservationBuilder().build_patient_observation(
        _rich_state()
    )["patient_internal_state"]

    assert patient_internal_state["hidden_history"] == [HIDDEN_HISTORY]
    assert patient_internal_state["hidden_allergies"] == ["secret allergy"]
    assert patient_internal_state["hidden_home_medications"] == ["secret medication"]


def test_patient_observation_includes_patient_emotion() -> None:
    observation = ObservationBuilder().build_patient_observation(_rich_state())

    assert observation["patient_emotion"]["label"] == "worried"
    assert observation["patient_emotion"]["intensity"] == "medium"


def test_patient_observation_does_not_include_full_test_bank() -> None:
    observation = ObservationBuilder().build_patient_observation(_rich_state())

    assert "test_bank" not in observation
    assert "available_diagnostic_tests" not in observation


def test_patient_observation_does_not_include_known_facts_as_a_whole() -> None:
    observation = ObservationBuilder().build_patient_observation(_rich_state())

    assert "known_facts" not in observation
    assert DISCLOSED_HISTORY not in _observation_text(observation)


def test_patient_observation_does_not_include_unreleased_diagnostic_results() -> None:
    observation = ObservationBuilder().build_patient_observation(_rich_state())

    text = _observation_text(observation)
    assert UNRELEASED_ECG_RESULT not in text
    assert UNRELEASED_TROPONIN_RESULT not in text


def test_relative_observation_does_not_include_hidden_diagnostic_results() -> None:
    observation = ObservationBuilder().build_relative_observation(_rich_state())

    text = _observation_text(observation)
    assert UNRELEASED_ECG_RESULT not in text
    assert UNRELEASED_TROPONIN_RESULT not in text


def test_relative_observation_does_not_include_full_numeric_vitals() -> None:
    observation = ObservationBuilder().build_relative_observation(_rich_state())
    visible_status = observation["visible_patient_status"]

    assert "vitals" not in visible_status
    assert visible_status["status_flags"] == {
        "is_alive": True,
        "can_speak": True,
        "is_conscious": True,
    }
    assert visible_status["visible_features"] == {
        "work_of_breathing": "increased"
    }
    text = _observation_text(visible_status)
    for vital_key in ("HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"):
        assert vital_key not in text


def test_private_clinician_nurse_messages_are_hidden_from_patient_and_relative() -> None:
    state = GlobalState(
        runtime_state={
            "messages": [
                {
                    "speaker": "clinician",
                    "recipient": "nurse",
                    "content": "PRIVATE TEAM MESSAGE",
                },
                {
                    "speaker": "clinician",
                    "recipient": None,
                    "content": "PUBLIC MESSAGE",
                },
            ]
        }
    )
    builder = ObservationBuilder()

    patient_observation = builder.build_patient_observation(state)
    relative_observation = builder.build_relative_observation(state)

    assert "PRIVATE TEAM MESSAGE" not in _observation_text(patient_observation)
    assert "PRIVATE TEAM MESSAGE" not in _observation_text(relative_observation)
    assert "PUBLIC MESSAGE" in _observation_text(patient_observation)
    assert "PUBLIC MESSAGE" in _observation_text(relative_observation)


def test_observation_builder_does_not_mutate_global_state() -> None:
    state = _rich_state()
    before = deepcopy(state.model_dump())

    ObservationBuilder().build_for(
        ["clinician", "nurse", "patient", "relative"],
        state,
    )

    assert state.model_dump() == before


def test_mutating_returned_observation_does_not_mutate_global_state() -> None:
    state = _rich_state()
    observation = ObservationBuilder().build_patient_observation(state)

    observation["patient_internal_state"]["symptoms"].append("mutated symptom")
    observation["communication_ability"]["can_speak"] = False

    assert state.truth_state.patient_internal_state.symptoms == [
        "dyspnea",
        "chest tightness",
    ]
    assert state.patient_state.status_flags.can_speak is True


def test_observation_builder_does_not_import_transition_engines() -> None:
    source = inspect.getsource(observation_builder_module)

    assert "transition_engines" not in source


def test_observation_builder_does_not_call_state_manager() -> None:
    source = inspect.getsource(observation_builder_module)

    assert "StateManager" not in source
    assert "state_manager" not in source


def test_last_turn_events_appear_in_clinician_and_nurse_observations() -> None:
    builder = ObservationBuilder()
    state = _rich_state()

    clinician_observation = builder.build_clinician_observation(state)
    nurse_observation = builder.build_nurse_observation(state)

    assert clinician_observation["last_turn_events"][0]["type"] == "diagnostic_release"
    assert nurse_observation["last_turn_events"][0]["type"] == "diagnostic_release"


def test_current_turn_events_are_not_included_in_observations() -> None:
    state = GlobalState(
        runtime_state={
            "last_turn_events": [
                {
                    "type": "diagnostic_release",
                    "payload": {"marker": "LAST TURN EVENT MARKER"},
                }
            ],
            "current_turn_events": [
                {
                    "type": "diagnostic_release",
                    "payload": {"marker": "CURRENT TURN INTERNAL MARKER"},
                }
            ],
        }
    )

    observations = ObservationBuilder().build_for(
        ["clinician", "nurse", "patient", "relative"],
        state,
    )

    text = _observation_text(observations)
    assert "LAST TURN EVENT MARKER" in text
    assert "CURRENT TURN INTERNAL MARKER" not in text


def test_last_bedside_event_if_any_is_derived_from_last_turn_events() -> None:
    state = GlobalState(
        runtime_state={
            "last_turn_events": [
                {
                    "type": "nurse_shadow_execution",
                    "payload": {"marker": "LAST BEDSIDE EVENT"},
                }
            ],
            "current_turn_events": [
                {
                    "type": "nurse_shadow_execution",
                    "payload": {"marker": "CURRENT BEDSIDE EVENT"},
                }
            ],
        }
    )

    observation = ObservationBuilder().build_nurse_observation(state)

    assert "current_bedside_event_if_any" not in observation
    assert observation["last_bedside_event_if_any"]["payload"]["marker"] == (
        "LAST BEDSIDE EVENT"
    )


def test_clinician_pending_questions_are_scoped_to_clinician_required_unresolved() -> None:
    state = GlobalState(
        runtime_state={
            "pending_questions": [
                {
                    "source_agent": "clinician",
                    "target_agent": "patient",
                    "question_text": "Clinician asked patient.",
                    "requires_response": True,
                    "is_resolved": False,
                },
                {
                    "source_agent": "patient",
                    "target_agent": "clinician",
                    "question_text": "Patient asked clinician.",
                    "requires_response": True,
                    "is_resolved": False,
                },
                {
                    "source_agent": "nurse",
                    "target_agent": "patient",
                    "question_text": "Nurse asked patient.",
                    "requires_response": True,
                    "is_resolved": False,
                },
                {
                    "source_agent": "clinician",
                    "target_agent": "relative",
                    "question_text": "Resolved clinician question.",
                    "requires_response": True,
                    "is_resolved": True,
                },
                {
                    "source_agent": "clinician",
                    "target_agent": "nurse",
                    "question_text": "Non-required clinician question.",
                    "requires_response": False,
                    "is_resolved": False,
                },
            ]
        }
    )

    observation = ObservationBuilder().build_clinician_observation(state)

    assert [
        question["question_text"] for question in observation["pending_questions"]
    ] == [
        "Clinician asked patient.",
        "Patient asked clinician.",
    ]


def test_newly_available_results_appear_as_per_turn_field_where_appropriate() -> None:
    builder = ObservationBuilder()
    state = _rich_state()

    clinician_observation = builder.build_clinician_observation(state)
    nurse_observation = builder.build_nurse_observation(state)

    expected = [{"name": "VBG", "result": "released pH 7.30"}]
    assert clinician_observation["newly_available_results"] == expected
    assert nurse_observation["newly_available_results"] == expected


def test_missing_optional_fields_produce_empty_lists_or_none_not_errors() -> None:
    builder = ObservationBuilder()
    state = GlobalState()

    clinician_observation = builder.build_clinician_observation(state)
    nurse_observation = builder.build_nurse_observation(state)
    patient_observation = builder.build_patient_observation(state)
    relative_observation = builder.build_relative_observation(state)

    assert clinician_observation["available_diagnostic_tests"] == []
    assert clinician_observation["newly_available_results"] == []
    assert clinician_observation["recent_messages"] == []
    assert clinician_observation["pending_questions"] == []
    assert clinician_observation["last_turn_events"] == []
    assert "safe_case_context" not in clinician_observation
    assert nurse_observation["last_bedside_event_if_any"] is None
    assert patient_observation["patient_internal_state"]["chief_complaint"] is None
    assert patient_observation["patient_internal_state"]["symptoms"] == []
    assert patient_observation["patient_internal_state"]["hidden_history"] == []
    assert patient_observation["patient_internal_state"]["hidden_allergies"] == []
    assert patient_observation["patient_internal_state"][
        "hidden_home_medications"
    ] == []
    assert patient_observation["patient_internal_state"]["disclosure_rules"] is None
    assert patient_observation["perceived_bedside_actions"] == []
    assert relative_observation["visible_last_turn_events"] == []
    assert relative_observation["family_side_hidden_info"] is None


def test_observations_do_not_include_raw_text() -> None:
    state = GlobalState(
        runtime_state={
            "newly_available_results": [
                DiagnosticResult(name="ECG", result="released atrial fibrillation")
            ],
            "last_turn_events": [
                Event(
                    type="diagnostic_release",
                    payload={
                        "test_name": "ECG",
                        "raw_text": "legacy top-level event text",
                        "nested": {"raw_text": "legacy nested event text"},
                    },
                )
            ],
        }
    )

    observations = ObservationBuilder().build_for(
        ["clinician", "nurse", "patient", "relative"],
        state,
    )

    assert "raw_text" not in _observation_text(observations)
