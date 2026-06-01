from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.facts.schemas import (
    FactExtractionResult,
    KnownAllergy,
    KnownHistory,
    KnownMedication,
    KnownSymptom,
)
from ed_world_model.state.global_state import (
    DiagnosticResult,
    Event,
    GlobalState,
    Message,
    PatientEmotion,
    PatientState,
    PendingDiagnosticResult,
    PendingQuestion,
    StatusFlags,
    Vitals,
)
from ed_world_model.state.state_manager import StateManager


def state_with_test_bank() -> GlobalState:
    return GlobalState(
        truth_state={
            "test_bank": [
                {
                    "name": "ECG",
                    "result": "LVH and A.fib",
                    "turnaround_turns": 1,
                },
                {
                    "name": "Chest X-ray",
                    "result": "CHF",
                    "turnaround_turns": 3,
                },
                {
                    "name": "VBG",
                    "result": "pH 7.30, PCO2 35",
                },
                {
                    "name": "Point-of-care glucose",
                    "result": "94 mg/dL",
                    "turnaround_turns": 0,
                },
            ]
        }
    )


def state_with_rich_physiology() -> GlobalState:
    return GlobalState(
        truth_state={
            "scenario_description": "Hidden scenario",
            "test_bank": [
                {
                    "name": "ECG",
                    "result": "LVH and A.fib",
                    "turnaround_turns": 1,
                }
            ],
        },
        patient_state={
            "vitals": {
                "HR": 110,
                "BP_sys": 150,
                "BP_dia": 95,
                "RR": 34,
                "O2Sat": 80,
                "T": 36.9,
            },
            "features": {
                "rhythm": "NSR",
                "oxygen_device": None,
                "FiO2": 0.21,
                "intubated": False,
                "vent": False,
                "PEEP_cmH2O": None,
                "neuro_status": "GCS 15",
                "glucose_mmol_l": 6.2,
            },
            "status_flags": {
                "is_alive": True,
                "can_speak": True,
                "is_conscious": True,
            },
        },
        known_facts={
            "known_history": [
                {
                    "item": "hypertension",
                    "status": "present",
                    "source_texts": ["I have hypertension."],
                }
            ],
            "available_results": [{"name": "ECG", "result": "LVH and A.fib"}],
        },
        runtime_state={
            "messages": [{"speaker": "patient", "content": "Help."}],
            "pending_diagnostic_results": [
                {"test_name": "ECG", "ordered_at_turn": 0, "ready_at_turn": 1}
            ],
            "newly_available_results": [
                {"name": "ECG", "result": "LVH and A.fib"}
            ],
        },
    )


def test_add_message_appends_only_to_runtime_messages() -> None:
    manager = StateManager()

    manager.add_message(Message(speaker="patient", content="I feel short of breath."))

    assert manager.state.runtime_state.messages == [
        Message(speaker="patient", content="I feel short of breath.")
    ]
    assert manager.state.runtime_state.last_turn_events == []
    assert manager.state.runtime_state.current_turn_events == []


def test_record_event_appends_only_to_current_turn_events() -> None:
    manager = StateManager()

    manager.record_event(Event(type="validation_drop", payload={"reason": "invalid"}))

    assert manager.state.runtime_state.current_turn_events == [
        Event(type="validation_drop", payload={"reason": "invalid"})
    ]
    assert manager.state.runtime_state.last_turn_events == []
    assert manager.state.runtime_state.messages == []


def test_update_known_facts_updates_known_facts_without_touching_truth_state() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "hidden_history": ["diabetes"],
                "hidden_allergies": ["penicillin"],
            }
        }
    )
    truth_before = state.truth_state.model_dump()
    manager = StateManager(state)

    manager.update_known_facts(
        known_history=[
            KnownHistory(
                item="hypertension",
                status="present",
                source_texts=["I have hypertension."],
            )
        ],
        known_symptoms=KnownSymptom(
            name="shortness of breath",
            status="present",
            source_texts=["I feel short of breath."],
        ),
    )

    assert manager.state.known_facts.known_history == [
        KnownHistory(
            item="hypertension",
            status="present",
            source_texts=["I have hypertension."],
        )
    ]
    assert manager.state.known_facts.known_symptoms == [
        KnownSymptom(
            name="shortness of breath",
            status="present",
            source_texts=["I feel short of breath."],
        )
    ]
    assert manager.state.known_facts.known_allergies == []
    assert manager.state.truth_state.model_dump() == truth_before
    assert all(
        history.item != "diabetes"
        for history in manager.state.known_facts.known_history
    )


def test_apply_fact_extraction_result_creates_structured_symptom() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            symptoms=[
                KnownSymptom(
                    name="shortness of breath",
                    status="present",
                    source_texts=["I feel short of breath."],
                )
            ]
        )
    )

    assert manager.state.known_facts.known_symptoms == [
        KnownSymptom(
            name="shortness of breath",
            status="present",
            source_texts=["I feel short of breath."],
        )
    ]


def test_apply_fact_extraction_result_merges_symptom_onset_and_severity() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        {
            "symptoms": [
                {
                    "name": "Shortness   of Breath",
                    "status": "uncertain",
                    "source_texts": ["I am short of breath."],
                }
            ]
        }
    )
    manager.apply_fact_extraction_result(
        {
            "symptoms": [
                {
                    "name": "shortness-of-breath",
                    "status": "present",
                    "onset": "a few hours ago",
                    "severity": "severe",
                    "source_texts": [
                        "I am short of breath.",
                        "It started a few hours ago.",
                    ],
                }
            ]
        }
    )

    symptoms = manager.state.known_facts.known_symptoms
    assert len(symptoms) == 1
    assert symptoms[0].name == "Shortness   of Breath"
    assert symptoms[0].status == "present"
    assert symptoms[0].onset == "a few hours ago"
    assert symptoms[0].severity == "severe"
    assert symptoms[0].source_texts == [
        "I am short of breath.",
        "It started a few hours ago.",
    ]


def test_apply_fact_extraction_result_does_not_use_synonym_matching() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        {
            "symptoms": [
                {
                    "name": "SOB",
                    "status": "present",
                    "source_texts": ["I have SOB."],
                },
                {
                    "name": "shortness of breath",
                    "status": "present",
                    "source_texts": ["I am short of breath."],
                },
            ]
        }
    )

    assert [symptom.name for symptom in manager.state.known_facts.known_symptoms] == [
        "SOB",
        "shortness of breath",
    ]


def test_apply_fact_extraction_result_preserves_conflicting_symptom_status() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        {
            "symptoms": [
                {
                    "name": "chest pain",
                    "status": "present",
                    "source_texts": ["I have chest pain."],
                }
            ]
        }
    )
    manager.apply_fact_extraction_result(
        {
            "symptoms": [
                {
                    "name": "chest pain",
                    "status": "absent",
                    "source_texts": ["I do not have chest pain now."],
                }
            ]
        }
    )

    symptoms = manager.state.known_facts.known_symptoms
    assert len(symptoms) == 1
    assert symptoms[0].status == "present"
    assert symptoms[0].source_texts == [
        "I have chest pain.",
        "I do not have chest pain now.",
    ]


def test_apply_fact_extraction_result_appends_non_symptom_structured_facts() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            history=[
                KnownHistory(
                    item="asthma",
                    status="present",
                    source_texts=["He has asthma."],
                )
            ],
            allergies=[
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    reaction="hives",
                    source_texts=["Penicillin gives her hives."],
                )
            ],
            medications=[
                KnownMedication(
                    name="albuterol",
                    status="current",
                    source_texts=["She uses albuterol."],
                )
            ],
        )
    )

    assert manager.state.known_facts.known_history[0].item == "asthma"
    assert manager.state.known_facts.known_allergies[0].reaction == "hives"
    assert manager.state.known_facts.known_medications[0].name == "albuterol"


def test_history_repeated_disclosure_merges_source_texts() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            history=[
                KnownHistory(
                    item="asthma",
                    status="present",
                    source_texts=["He has asthma."],
                )
            ]
        )
    )
    manager.apply_fact_extraction_result(
        FactExtractionResult(
            history=[
                KnownHistory(
                    item="asthma",
                    status="present",
                    source_texts=["Yeah, asthma since childhood."],
                )
            ]
        )
    )

    assert manager.state.known_facts.known_history == [
        KnownHistory(
            item="asthma",
            status="present",
            source_texts=[
                "He has asthma.",
                "Yeah, asthma since childhood.",
            ],
        )
    ]


def test_allergy_repeated_disclosure_merges_source_texts() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            allergies=[
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    source_texts=["Penicillin gives her hives."],
                )
            ]
        )
    )
    manager.apply_fact_extraction_result(
        FactExtractionResult(
            allergies=[
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    source_texts=["She is allergic to penicillin."],
                )
            ]
        )
    )

    assert len(manager.state.known_facts.known_allergies) == 1
    assert manager.state.known_facts.known_allergies[0].source_texts == [
        "Penicillin gives her hives.",
        "She is allergic to penicillin.",
    ]


def test_medication_repeated_disclosure_merges_source_texts() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            medications=[
                KnownMedication(
                    name="albuterol",
                    status="current",
                    source_texts=["She uses albuterol."],
                )
            ]
        )
    )
    manager.apply_fact_extraction_result(
        FactExtractionResult(
            medications=[
                KnownMedication(
                    name="albuterol",
                    status="current",
                    source_texts=["Yes, albuterol at home."],
                )
            ]
        )
    )

    assert manager.state.known_facts.known_medications == [
        KnownMedication(
            name="albuterol",
            status="current",
            source_texts=["She uses albuterol.", "Yes, albuterol at home."],
        )
    ]


def test_non_symptom_facts_with_different_status_remain_separate() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            history=[
                KnownHistory(
                    item="asthma",
                    status="present",
                    source_texts=["He has asthma."],
                ),
                KnownHistory(
                    item="asthma",
                    status="uncertain",
                    source_texts=["I am not sure about asthma."],
                ),
            ],
            medications=[
                KnownMedication(
                    name="metformin",
                    status="current",
                    source_texts=["She takes metformin."],
                ),
                KnownMedication(
                    name="metformin",
                    status="not_taking",
                    source_texts=["She stopped metformin."],
                ),
            ],
        )
    )

    assert len(manager.state.known_facts.known_history) == 2
    assert len(manager.state.known_facts.known_medications) == 2


def test_allergies_with_different_reactions_remain_separate() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            allergies=[
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    reaction="hives",
                    source_texts=["Penicillin gives her hives."],
                ),
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    reaction="anaphylaxis",
                    source_texts=["Penicillin caused anaphylaxis."],
                ),
            ]
        )
    )

    assert len(manager.state.known_facts.known_allergies) == 2


def test_allergy_reaction_normalization_merges_matching_reactions() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            allergies=[
                KnownAllergy(
                    substance="penicillin",
                    status="present",
                    reaction="severe   hives",
                    source_texts=["Penicillin caused severe hives."],
                ),
                KnownAllergy(
                    substance="PENICILLIN",
                    status="present",
                    reaction=" severe hives ",
                    source_texts=["Yes, severe hives from penicillin."],
                ),
            ]
        )
    )

    assert len(manager.state.known_facts.known_allergies) == 1
    assert manager.state.known_facts.known_allergies[0].source_texts == [
        "Penicillin caused severe hives.",
        "Yes, severe hives from penicillin.",
    ]


def test_non_symptom_source_text_exact_duplicates_are_not_duplicated() -> None:
    manager = StateManager()

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            history=[
                KnownHistory(
                    item="asthma",
                    status="present",
                    source_texts=["He has asthma."],
                ),
                KnownHistory(
                    item="ASTHMA",
                    status="present",
                    source_texts=["He has asthma."],
                ),
            ]
        )
    )

    assert manager.state.known_facts.known_history == [
        KnownHistory(
            item="asthma",
            status="present",
            source_texts=["He has asthma."],
        )
    ]


def test_apply_fact_extraction_result_does_not_copy_hidden_truth() -> None:
    state = GlobalState(
        truth_state={
            "patient_internal_state": {
                "symptoms": ["hidden fever"],
                "hidden_history": ["hidden diabetes"],
            }
        }
    )
    manager = StateManager(state)

    manager.apply_fact_extraction_result(
        FactExtractionResult(
            symptoms=[
                KnownSymptom(
                    name="dizziness",
                    status="present",
                    source_texts=["I feel dizzy."],
                )
            ]
        )
    )

    dumped = manager.state.known_facts.model_dump()
    assert "hidden fever" not in str(dumped)
    assert "hidden diabetes" not in str(dumped)


def test_create_pending_diagnostic_result_creates_pending_for_existing_test() -> None:
    manager = StateManager(state_with_test_bank())

    pending = manager.create_pending_diagnostic_result("ECG")

    assert pending == PendingDiagnosticResult(
        test_name="ECG",
        ordered_at_turn=0,
        ready_at_turn=1,
    )
    assert manager.state.runtime_state.pending_diagnostic_results == [pending]
    assert manager.state.known_facts.available_results == []
    assert manager.state.runtime_state.current_turn_events[-1].type == (
        "diagnostic_order_created"
    )


def test_create_pending_diagnostic_result_uses_test_turnaround_turns() -> None:
    manager = StateManager(state_with_test_bank())

    pending = manager.create_pending_diagnostic_result("Chest X-ray", current_turn=4)

    assert pending.ordered_at_turn == 4
    assert pending.ready_at_turn == 7


def test_create_pending_diagnostic_result_uses_default_turnaround_when_missing() -> None:
    manager = StateManager(state_with_test_bank())

    pending = manager.create_pending_diagnostic_result("VBG", current_turn=4)

    assert pending.ready_at_turn == 4 + DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS


def test_create_pending_diagnostic_result_honors_zero_turnaround() -> None:
    manager = StateManager(state_with_test_bank())

    pending = manager.create_pending_diagnostic_result(
        "Point-of-care glucose",
        current_turn=4,
    )

    assert pending.ordered_at_turn == 4
    assert pending.ready_at_turn == 4


def test_create_pending_diagnostic_result_raises_clear_error_for_unknown_test() -> None:
    manager = StateManager(state_with_test_bank())

    with pytest.raises(ValueError, match="Unknown diagnostic test_name 'Troponin'"):
        manager.create_pending_diagnostic_result("Troponin")


def test_release_ready_diagnostic_results_releases_only_ready_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)

    released = manager.release_ready_diagnostic_results(current_turn=1)

    assert released == [DiagnosticResult(name="ECG", result="LVH and A.fib")]
    assert manager.state.runtime_state.pending_diagnostic_results == [
        PendingDiagnosticResult(
            test_name="Chest X-ray",
            ordered_at_turn=0,
            ready_at_turn=3,
        )
    ]


def test_release_ready_diagnostic_results_releases_multiple_ready_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("Point-of-care glucose", current_turn=0)
    manager.create_pending_diagnostic_result("ECG", current_turn=0)
    manager.create_pending_diagnostic_result("VBG", current_turn=0)
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)

    released = manager.release_ready_diagnostic_results(current_turn=2)

    assert released == [
        DiagnosticResult(name="Point-of-care glucose", result="94 mg/dL"),
        DiagnosticResult(name="ECG", result="LVH and A.fib"),
        DiagnosticResult(name="VBG", result="pH 7.30, PCO2 35"),
    ]
    assert manager.state.runtime_state.pending_diagnostic_results == [
        PendingDiagnosticResult(
            test_name="Chest X-ray",
            ordered_at_turn=0,
            ready_at_turn=3,
        )
    ]


def test_released_diagnostic_result_enters_known_facts_available_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=1)

    assert manager.state.known_facts.available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]


def test_released_diagnostic_result_enters_newly_available_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=1)

    assert manager.state.runtime_state.newly_available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]


def test_unreleased_diagnostic_result_does_not_enter_available_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=2)

    assert manager.state.known_facts.available_results == []
    assert manager.state.runtime_state.newly_available_results == []


def test_released_diagnostic_result_is_removed_from_pending_results() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=1)

    assert manager.state.runtime_state.pending_diagnostic_results == []


def test_diagnostic_release_event_payload_includes_released_at_turn() -> None:
    manager = StateManager(state_with_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=1)

    release_events = [
        event
        for event in manager.state.runtime_state.current_turn_events
        if event.type == "diagnostic_release"
    ]
    assert release_events[-1].payload["released_at_turn"] == 1


def test_advance_turn_increments_turn_index() -> None:
    manager = StateManager(GlobalState(runtime_state={"turn_index": 4}))

    new_turn = manager.advance_turn()

    assert new_turn == 5
    assert manager.state.runtime_state.turn_index == 5


def test_advance_turn_moves_current_events_to_last_events_and_clears_current() -> None:
    manager = StateManager(
        GlobalState(
            runtime_state={
                "newly_available_results": [
                    {"name": "ECG", "result": "LVH and A.fib"}
                ],
                "last_turn_events": [
                    {"type": "previous_event", "payload": {"turn": -1}}
                ],
                "current_turn_events": [
                    {"type": "diagnostic_release", "payload": {"test_name": "ECG"}}
                ],
            }
        )
    )

    manager.advance_turn()

    assert manager.state.runtime_state.newly_available_results == []
    assert manager.state.runtime_state.last_turn_events == [
        Event(type="diagnostic_release", payload={"test_name": "ECG"})
    ]
    assert manager.state.runtime_state.current_turn_events == []


def test_last_turn_events_remain_visible_after_advance_turn() -> None:
    manager = StateManager()
    event = Event(type="nurse_shadow_execution", payload={"executed_by": "nurse"})

    manager.record_event(event)
    manager.advance_turn()

    assert manager.state.runtime_state.last_turn_events == [event]
    assert manager.state.runtime_state.current_turn_events == []


def test_advance_turn_preserves_longer_lived_runtime_and_known_fields() -> None:
    manager = StateManager(
        GlobalState(
            known_facts={
                "available_results": [
                    {"name": "ECG", "result": "LVH and A.fib"}
                ]
            },
            runtime_state={
                "messages": [
                    {"speaker": "patient", "content": "I feel short of breath."}
                ],
                "pending_diagnostic_results": [
                    {
                        "test_name": "Chest X-ray",
                        "ordered_at_turn": 0,
                        "ready_at_turn": 3,
                    }
                ],
                "pending_questions": [
                    {
                        "source_agent": "clinician",
                        "target_agent": "patient",
                        "question_text": "Do you have chest pain?",
                    }
                ],
                "required_response_agents": ["patient"],
            },
        )
    )

    manager.advance_turn()

    assert manager.state.runtime_state.messages == [
        Message(speaker="patient", content="I feel short of breath.")
    ]
    assert manager.state.known_facts.available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]
    assert manager.state.runtime_state.pending_diagnostic_results == [
        PendingDiagnosticResult(
            test_name="Chest X-ray",
            ordered_at_turn=0,
            ready_at_turn=3,
        )
    ]
    assert manager.state.runtime_state.pending_questions == [
        PendingQuestion(
            source_agent="clinician",
            target_agent="patient",
            question_text="Do you have chest pain?",
        )
    ]
    assert manager.state.runtime_state.required_response_agents == ["patient"]


def test_resolve_pending_questions_can_skip_same_turn_questions() -> None:
    manager = StateManager(
        GlobalState(
            runtime_state={
                "turn_index": 3,
                "pending_questions": [
                    {
                        "source_agent": "clinician",
                        "target_agent": "patient",
                        "question_text": "Old question?",
                        "created_at_turn": 2,
                    },
                    {
                        "source_agent": "clinician",
                        "target_agent": "patient",
                        "question_text": "Same-turn question?",
                        "created_at_turn": 3,
                    },
                ],
                "required_response_agents": ["patient"],
            }
        )
    )

    resolved = manager.resolve_pending_questions_for_agent(
        "patient",
        created_before_turn=3,
    )

    assert [question.question_text for question in resolved] == ["Old question?"]
    assert manager.state.runtime_state.pending_questions[0].is_resolved is True
    assert manager.state.runtime_state.pending_questions[1].is_resolved is False
    assert manager.state.runtime_state.required_response_agents == ["patient"]


def test_apply_physiology_update_updates_patient_state_only() -> None:
    state = GlobalState(
        truth_state={"scenario_description": "Hidden scenario"},
        known_facts={
            "known_history": [
                {
                    "item": "hypertension",
                    "status": "present",
                    "source_texts": ["I have hypertension."],
                }
            ]
        },
        psych_state={"patient_emotion": {"label": "anxious", "intensity": "medium"}},
        runtime_state={
            "messages": [{"speaker": "patient", "content": "Help."}],
            "last_turn_events": [{"type": "diagnostic_release"}],
        },
    )
    manager = StateManager(state)
    truth_before = state.truth_state.model_dump()
    known_before = state.known_facts.model_dump()
    psych_before = state.psych_state.model_dump()
    runtime_before = state.runtime_state.model_dump()

    manager.apply_physiology_update(
        PatientState(
            vitals=Vitals(HR=120, BP_sys=100, BP_dia=60, RR=24, O2Sat=92, T=37.2),
            status_flags=StatusFlags(is_alive=True, can_speak=False),
        )
    )

    assert manager.state.patient_state.vitals.HR == 120
    assert manager.state.patient_state.status_flags.can_speak is False
    assert manager.state.truth_state.model_dump() == truth_before
    assert manager.state.known_facts.model_dump() == known_before
    assert manager.state.psych_state.model_dump() == psych_before
    assert manager.state.runtime_state.model_dump() == runtime_before


def test_apply_physiology_update_merges_partial_features() -> None:
    manager = StateManager(state_with_rich_physiology())

    manager.apply_physiology_update(
        features={
            "oxygen_device": "NRB",
            "FiO2": 0.8,
        }
    )

    assert manager.state.patient_state.features.model_dump() == {
        "rhythm": "NSR",
        "oxygen_device": "NRB",
        "FiO2": 0.8,
        "intubated": False,
        "vent": False,
        "PEEP_cmH2O": None,
        "neuro_status": "GCS 15",
        "glucose_mmol_l": 6.2,
    }


def test_apply_physiology_update_merges_partial_vitals() -> None:
    manager = StateManager(state_with_rich_physiology())

    manager.apply_physiology_update(vitals={"HR": 98, "O2Sat": 94})

    assert manager.state.patient_state.vitals == Vitals(
        HR=98,
        BP_sys=150,
        BP_dia=95,
        RR=34,
        O2Sat=94,
        T=36.9,
    )
    assert manager.state.patient_state.features.model_dump()["rhythm"] == "NSR"
    assert manager.state.patient_state.status_flags.can_speak is True


def test_apply_physiology_update_merges_partial_status_flags() -> None:
    manager = StateManager(state_with_rich_physiology())

    manager.apply_physiology_update(status_flags={"can_speak": False})

    assert manager.state.patient_state.status_flags == StatusFlags(
        is_alive=True,
        can_speak=False,
        is_conscious=True,
    )
    assert manager.state.patient_state.vitals.HR == 110
    assert manager.state.patient_state.features.model_dump()["rhythm"] == "NSR"


def test_apply_physiology_update_explicit_none_overwrites_existing_value() -> None:
    manager = StateManager(
        GlobalState(
            patient_state={
                "vitals": {"O2Sat": 95},
                "features": {"oxygen_device": "NRB"},
            }
        )
    )

    manager.apply_physiology_update(
        vitals={"O2Sat": None},
        features={"oxygen_device": None},
    )

    assert manager.state.patient_state.vitals.O2Sat is None
    assert manager.state.patient_state.features.model_dump()["oxygen_device"] is None


def test_partial_physiology_update_preserves_non_patient_state() -> None:
    state = state_with_rich_physiology()
    manager = StateManager(state)
    truth_before = state.truth_state.model_dump()
    known_before = state.known_facts.model_dump()
    runtime_before = state.runtime_state.model_dump()

    manager.apply_physiology_update(features={"oxygen_device": "NRB"})

    assert manager.state.truth_state.model_dump() == truth_before
    assert manager.state.known_facts.model_dump() == known_before
    assert manager.state.runtime_state.model_dump() == runtime_before


def test_apply_physiology_update_records_event_when_requested() -> None:
    manager = StateManager()

    manager.apply_physiology_update(vitals={"HR": 110}, record_event=True)

    assert manager.state.runtime_state.current_turn_events == [
        Event(
            type="physiology_update",
            turn_index=0,
            payload={"updated_fields": ["vitals"]},
        )
    ]
    assert manager.state.runtime_state.last_turn_events == []


def test_apply_physiology_update_record_event_false_does_not_append_event() -> None:
    manager = StateManager()

    manager.apply_physiology_update(vitals={"HR": 95}, record_event=False)

    assert manager.state.runtime_state.current_turn_events == []


def test_apply_physiology_update_rejects_neither_and_both_inputs() -> None:
    manager = StateManager()

    with pytest.raises(ValueError, match="No physiology update was provided"):
        manager.apply_physiology_update()

    with pytest.raises(ValueError, match="either patient_state or individual"):
        manager.apply_physiology_update(PatientState(), vitals={"HR": 95})


def test_update_patient_emotion_updates_patient_emotion_only() -> None:
    state = GlobalState(
        truth_state={"scenario_description": "Hidden scenario"},
        patient_state={"vitals": {"HR": 90}},
        known_facts={
            "known_history": [
                {
                    "item": "hypertension",
                    "status": "present",
                    "source_texts": ["I have hypertension."],
                }
            ]
        },
        runtime_state={
            "messages": [{"speaker": "patient", "content": "Help."}],
            "last_turn_events": [{"type": "physiology_update"}],
        },
    )
    manager = StateManager(state)
    truth_before = state.truth_state.model_dump()
    patient_before = state.patient_state.model_dump()
    known_before = state.known_facts.model_dump()
    runtime_before = state.runtime_state.model_dump()

    manager.update_patient_emotion(PatientEmotion(label="fearful", intensity="high"))

    assert manager.state.psych_state.patient_emotion == PatientEmotion(
        label="fearful",
        intensity="high",
    )
    assert manager.state.truth_state.model_dump() == truth_before
    assert manager.state.patient_state.model_dump() == patient_before
    assert manager.state.known_facts.model_dump() == known_before
    assert manager.state.runtime_state.model_dump() == runtime_before


def test_update_patient_emotion_records_event_when_requested() -> None:
    manager = StateManager()

    manager.update_patient_emotion(label="calm", intensity="low", record_event=True)

    assert manager.state.runtime_state.current_turn_events == [
        Event(type="emotion_update", turn_index=0, payload={"label": "calm"})
    ]
    assert manager.state.runtime_state.last_turn_events == []


def test_update_patient_emotion_record_event_false_does_not_append_event() -> None:
    manager = StateManager()

    manager.update_patient_emotion(label="calm", record_event=False)

    assert manager.state.runtime_state.current_turn_events == []


def test_update_patient_emotion_rejects_neither_and_both_inputs() -> None:
    manager = StateManager()

    with pytest.raises(ValueError, match="No patient emotion update was provided"):
        manager.update_patient_emotion()

    with pytest.raises(ValueError, match="either patient_emotion or individual"):
        manager.update_patient_emotion(PatientEmotion(), label="calm")


def test_partial_update_patient_emotion_preserves_prior_fields() -> None:
    manager = StateManager(
        GlobalState(
            psych_state={
                "patient_emotion": {
                    "label": "fearful",
                    "intensity": "high",
                }
            }
        )
    )

    emotion = manager.update_patient_emotion(label="calm")

    assert emotion == PatientEmotion(
        label="calm",
        intensity="high",
    )


def test_state_manager_does_not_require_or_create_raw_text() -> None:
    manager = StateManager(state_with_test_bank())

    manager.add_message({"speaker": "patient", "content": "I feel short of breath."})
    manager.record_event({"type": "validation_drop", "payload": {"reason": "invalid"}})
    manager.create_pending_diagnostic_result("ECG")
    manager.release_ready_diagnostic_results(current_turn=1)
    manager.apply_physiology_update(vitals={"HR": 95})
    manager.update_patient_emotion(label="calm", intensity="low")

    dumped = manager.state.model_dump()
    assert "raw_text" not in str(dumped)
