from __future__ import annotations

from pathlib import Path
import sys

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.actions import (
    ResponseMode,
    ResponseOpportunity,
    ResponseOpportunityQueue,
    ResponseStatus,
    apply_response_to_discovered_memory,
    status_for_response_mode,
)
from ed_multiagent.memory import (
    ConversationMemory,
    DiscoveredClinicalMemory,
    GroundTruthMemory,
    PatientPrivateMemory,
    RelativePrivateMemory,
)


def _allergy_opportunity() -> ResponseOpportunity:
    return ResponseOpportunity(
        opportunity_id="question-1",
        asked_by="clinician",
        addressed_to="patient",
        question_text="Do you have any allergies?",
        question_type="allergy",
        requiredness="expected",
        sensitivity="low",
        urgency="routine",
        expected_slots=["allergies"],
        created_at_s=10,
    )


def test_ground_truth_allergy_is_not_visible_until_discovered():
    ground_truth = GroundTruthMemory(allergies=["penicillin"])
    discovered = DiscoveredClinicalMemory()

    assert ground_truth.allergies == ["penicillin"]
    assert discovered.allergies_known is False
    assert discovered.allergies == []


def test_discover_allergies_makes_allergies_clinician_visible():
    discovered = DiscoveredClinicalMemory()

    discovered.discover_allergies(["penicillin"])

    assert discovered.allergies_known is True
    assert discovered.allergies == ["penicillin"]


@pytest.mark.parametrize("addressed_to", ["patient", "relative"])
def test_response_opportunity_can_target_patient_or_relative(addressed_to: str):
    opportunity = ResponseOpportunity(
        opportunity_id=f"question-{addressed_to}",
        asked_by="clinician",
        addressed_to=addressed_to,
        question_text="What medications does the patient take?",
        question_type="medication",
        requiredness="expected",
        sensitivity="medium",
        urgency="routine",
        expected_slots=["medications"],
    )

    assert opportunity.asked_by == "clinician"
    assert opportunity.addressed_to == addressed_to
    assert opportunity.status == ResponseStatus.OPEN


def test_response_opportunity_queue_next_for_returns_next_open_for_role():
    queue = ResponseOpportunityQueue()
    patient_first = queue.add(
        ResponseOpportunity(
            opportunity_id="patient-1",
            addressed_to="patient",
            question_text="Where is the pain?",
            question_type="symptom",
            expected_slots=["symptom_history"],
            created_at_s=0,
        )
    )
    queue.add(
        ResponseOpportunity(
            opportunity_id="relative-1",
            addressed_to="relative",
            question_text="What happened today?",
            question_type="family_question",
            expected_slots=["recent_events"],
            created_at_s=1,
        )
    )
    queue.add(
        ResponseOpportunity(
            opportunity_id="patient-2",
            addressed_to="patient",
            question_text="Any allergies?",
            question_type="allergy",
            expected_slots=["allergies"],
            created_at_s=2,
        )
    )

    assert queue.next_for("patient") == patient_first

    queue.mark_resolved(patient_first.opportunity_id, ResponseStatus.ANSWERED)

    assert queue.next_for("patient").opportunity_id == "patient-2"
    assert queue.next_for("relative").opportunity_id == "relative-1"


def test_direct_answer_updates_discovered_memory_from_structured_slots():
    discovered = DiscoveredClinicalMemory()

    apply_response_to_discovered_memory(
        _allergy_opportunity(),
        ResponseMode.DIRECT_ANSWER,
        {"allergies": ["penicillin"]},
        discovered,
    )

    assert discovered.allergies_known is True
    assert discovered.allergies == ["penicillin"]


def test_partial_answer_updates_only_provided_slots():
    discovered = DiscoveredClinicalMemory()
    symptom_opportunity = ResponseOpportunity(
        opportunity_id="question-symptom",
        asked_by="clinician",
        addressed_to="patient",
        question_text="When did the pain start?",
        question_type="symptom",
        expected_slots=["allergies", "symptom_history"],
    )

    apply_response_to_discovered_memory(
        symptom_opportunity,
        ResponseMode.PARTIAL_ANSWER,
        {"symptom_history": {"onset": "45 minutes ago"}},
        discovered,
    )

    assert discovered.symptom_history == {"onset": "45 minutes ago"}
    assert discovered.allergies_known is False
    assert discovered.allergies == []
    assert discovered.medications_known is False
    assert discovered.pmh_known is False


def test_answer_slots_must_match_expected_slots_before_memory_update():
    discovered = DiscoveredClinicalMemory()

    with pytest.raises(ValueError, match="unexpected response slot"):
        apply_response_to_discovered_memory(
            _allergy_opportunity(),
            ResponseMode.DIRECT_ANSWER,
            {"medications": ["metformin"]},
            discovered,
        )

    assert discovered.medications_known is False
    assert discovered.medications == []


def test_empty_expected_slots_allows_supported_m1c_slots():
    discovered = DiscoveredClinicalMemory()
    opportunity = ResponseOpportunity(expected_slots=[])

    apply_response_to_discovered_memory(
        opportunity,
        ResponseMode.DIRECT_ANSWER,
        {
            "allergies": ["penicillin"],
            "medications": ["metformin"],
        },
        discovered,
    )

    assert discovered.allergies == ["penicillin"]
    assert discovered.medications == ["metformin"]


def test_vitals_snapshot_is_not_a_supported_m1c_slot_key():
    discovered = DiscoveredClinicalMemory()
    opportunity = ResponseOpportunity(expected_slots=[])

    with pytest.raises(ValueError, match="unsupported response slot"):
        apply_response_to_discovered_memory(
            opportunity,
            ResponseMode.DIRECT_ANSWER,
            {"vitals_snapshot": {"HR": 80}},
            discovered,
        )

    assert discovered.vitals_history == []


@pytest.mark.parametrize(
    "response_mode",
    [
        ResponseMode.REFUSE,
        ResponseMode.EVADE,
        ResponseMode.CLARIFY,
        ResponseMode.EMOTIONAL_REACTION,
    ],
)
def test_non_answer_modes_do_not_update_clinical_facts(
    response_mode: ResponseMode,
):
    discovered = DiscoveredClinicalMemory()

    apply_response_to_discovered_memory(
        _allergy_opportunity(),
        response_mode,
        {"allergies": ["penicillin"]},
        discovered,
    )

    assert discovered.allergies_known is False
    assert discovered.allergies == []


@pytest.mark.parametrize(
    "response_mode",
    [ResponseMode.SILENCE, ResponseMode.UNABLE],
)
def test_silence_and_unable_do_not_update_clinical_facts(
    response_mode: ResponseMode,
):
    discovered = DiscoveredClinicalMemory()

    apply_response_to_discovered_memory(
        _allergy_opportunity(),
        response_mode,
        {"medications": ["metformin"]},
        discovered,
    )

    assert discovered.medications_known is False
    assert discovered.medications == []


@pytest.mark.parametrize(
    "response_mode",
    [ResponseMode.CLARIFY, ResponseMode.EMOTIONAL_REACTION],
)
def test_non_closing_response_modes_leave_opportunity_open(
    response_mode: ResponseMode,
):
    opportunity = _allergy_opportunity()
    discovered = DiscoveredClinicalMemory()

    apply_response_to_discovered_memory(
        opportunity,
        response_mode,
        {"allergies": ["penicillin"]},
        discovered,
    )

    assert status_for_response_mode(response_mode) == ResponseStatus.OPEN
    assert opportunity.status == ResponseStatus.OPEN
    assert discovered.allergies == []


def test_conversation_memory_stores_raw_turns_and_maintains_recent_window_size():
    conversation = ConversationMemory(max_recent_turns=2)

    conversation.add_turn(
        speaker="clinician",
        target="patient",
        content="What brought you in?",
        time_s=0,
    )
    conversation.add_turn(
        speaker="patient",
        target="clinician",
        content="Chest pain.",
        time_s=5,
    )
    conversation.add_turn(
        speaker="clinician",
        target="patient",
        content="When did it start?",
        time_s=10,
    )

    assert len(conversation.raw_turns) == 3
    assert [turn.content for turn in conversation.recent_window] == [
        "Chest pain.",
        "When did it start?",
    ]


def test_clinical_facts_are_structured_not_only_in_conversation_memory():
    conversation = ConversationMemory()
    discovered = DiscoveredClinicalMemory()
    conversation.add_turn(
        speaker="patient",
        target="clinician",
        content="I am allergic to penicillin.",
        time_s=0,
    )

    assert discovered.allergies_known is False
    assert discovered.allergies == []

    apply_response_to_discovered_memory(
        _allergy_opportunity(),
        ResponseMode.DIRECT_ANSWER,
        {"allergies": ["penicillin"]},
        discovered,
    )

    assert conversation.raw_turns[0].content == "I am allergic to penicillin."
    assert discovered.allergies_known is True
    assert discovered.allergies == ["penicillin"]


def test_private_memories_are_distinct_from_ground_truth_memory():
    ground_truth = GroundTruthMemory(allergies=["penicillin"])
    patient_private = PatientPrivateMemory(
        symptom_knowledge={"location": "chest"},
        sensitive_topics=["substance_use"],
    )
    relative_private = RelativePrivateMemory(
        relationship="spouse",
        knows_allergies=False,
        private_facts={"arrival": "called EMS"},
    )

    assert ground_truth.allergies == ["penicillin"]
    assert patient_private.symptom_knowledge == {"location": "chest"}
    assert relative_private.private_facts == {"arrival": "called EMS"}
    assert relative_private.knows_allergies is False


def test_queue_can_expire_old_open_opportunities():
    queue = ResponseOpportunityQueue(
        [
            ResponseOpportunity(
                opportunity_id="old",
                addressed_to="patient",
                created_at_s=0,
            ),
            ResponseOpportunity(
                opportunity_id="fresh",
                addressed_to="patient",
                created_at_s=95,
            ),
        ]
    )

    expired = queue.expire_older_than(now_s=100, max_age_s=30)

    assert [opportunity.opportunity_id for opportunity in expired] == ["old"]
    assert queue.opportunities["old"].status == ResponseStatus.EXPIRED
    assert queue.opportunities["fresh"].status == ResponseStatus.OPEN


def test_queue_expires_opportunity_at_max_age_boundary():
    queue = ResponseOpportunityQueue(
        [
            ResponseOpportunity(
                opportunity_id="boundary",
                addressed_to="patient",
                created_at_s=0,
            ),
        ]
    )

    expired = queue.expire_older_than(now_s=10, max_age_s=10)

    assert [opportunity.opportunity_id for opportunity in expired] == ["boundary"]
    assert queue.opportunities["boundary"].status == ResponseStatus.EXPIRED


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        (ResponseMode.DIRECT_ANSWER, ResponseStatus.ANSWERED),
        (ResponseMode.PARTIAL_ANSWER, ResponseStatus.PARTIALLY_ANSWERED),
        (ResponseMode.REFUSE, ResponseStatus.REFUSED),
        (ResponseMode.EVADE, ResponseStatus.EVADED),
        (ResponseMode.SILENCE, ResponseStatus.IGNORED),
        (ResponseMode.UNABLE, ResponseStatus.UNABLE_TO_ANSWER),
    ],
)
def test_response_modes_map_to_resolution_statuses(
    mode: ResponseMode,
    status: ResponseStatus,
):
    assert status_for_response_mode(mode) == status
