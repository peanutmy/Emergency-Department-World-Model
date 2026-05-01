from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.actions import ResponseOpportunity, ResponseOpportunityQueue
from ed_multiagent.emsim_adapter import EngineSession
from ed_multiagent.memory import (
    ConversationMemory,
    DiscoveredClinicalMemory,
    GroundTruthMemory,
    PatientPrivateMemory,
    RelativePrivateMemory,
)
from ed_multiagent.observation import (
    ClinicianObservation,
    FORBIDDEN_OBSERVATION_KEYS,
    NurseObservation,
    ObservationGateway,
    SpeechCapacity,
    SubjectiveState,
)
from ed_multiagent.world import WorkflowEngine


NUMERIC_VITAL_KEYS = {"HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"}


def _base_state() -> dict:
    return {
        "vitals": {
            "HR": 120,
            "BP_sys": 92,
            "BP_dia": 55,
            "RR": 28,
            "O2Sat": 88,
            "T": 37.0,
        },
        "interventions": {
            "airway": False,
            "O2_device": None,
            "PEEP": 0,
            "FiO2": 0.21,
            "vent_rate": None,
            "vent_TV_ml": None,
            "intubated": False,
            "CPR_active": False,
            "defib_last_J": None,
            "pacing_active": False,
            "pacing_rate": None,
            "fluids_rate_ml_hr": 0,
            "fluid_type": None,
            "warming_active": False,
            "cooling_active": False,
            "needle_decompression": False,
            "chest_tube": False,
            "pericardiocentesis": False,
        },
        "mechanism": {
            "pathology": {
                "name": "septic_shock",
                "severity": "severe",
            }
        },
    }


def _subjective_state() -> SubjectiveState:
    return SubjectiveState(
        dyspnea_severity=0.8,
        confusion_level=0.2,
        palpitation=True,
        dizziness=0.4,
        pain_distress=0.6,
        speech_capacity=SpeechCapacity.SHORT_PHRASES,
        visible_distress=0.75,
    )


def _serialized(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _keys_in(value: Any) -> set[str]:
    value = _serialized(value)
    if isinstance(value, dict):
        found = set(value)
        for nested in value.values():
            found.update(_keys_in(nested))
        return found
    if isinstance(value, (list, tuple, set)):
        found: set[str] = set()
        for nested in value:
            found.update(_keys_in(nested))
        return found
    return set()


def _values_in(value: Any) -> list[Any]:
    value = _serialized(value)
    if isinstance(value, dict):
        values: list[Any] = []
        for nested in value.values():
            values.extend(_values_in(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for nested in value:
            values.extend(_values_in(nested))
        return values
    return [value]


def _gateway(**kwargs: Any) -> ObservationGateway:
    kwargs.setdefault("engine_session", EngineSession(_base_state()))
    return ObservationGateway(**kwargs)


def test_clinician_does_not_see_ground_truth_allergy_until_discovered():
    gateway = _gateway(
        ground_truth_memory=GroundTruthMemory(allergies=["penicillin"]),
        discovered_memory=DiscoveredClinicalMemory(),
    )

    observation = gateway.for_clinician()

    assert observation.allergies_known is False
    assert observation.allergies == []
    assert "penicillin" not in _values_in(observation)


def test_clinician_sees_discovered_allergy_after_discovery():
    discovered = DiscoveredClinicalMemory()
    discovered.discover_allergies(["penicillin"])

    observation = _gateway(discovered_memory=discovered).for_clinician()

    assert observation.allergies_known is True
    assert observation.allergies == ["penicillin"]


def test_clinician_does_not_see_ground_truth_diagnosis_or_pathology():
    gateway = _gateway(
        ground_truth_memory=GroundTruthMemory(
            diagnosis="anaphylaxis",
            pathology="distributive_shock",
        ),
        discovered_memory=DiscoveredClinicalMemory(),
    )

    observation = gateway.for_clinician()

    assert "anaphylaxis" not in _values_in(observation)
    assert "distributive_shock" not in _values_in(observation)
    assert "diagnosis" not in _keys_in(observation)
    assert "pathology" not in _keys_in(observation)


def test_clinician_does_not_receive_live_engine_vitals():
    state = _base_state()
    state["vitals"].update({"HR": 222, "O2Sat": 66})
    gateway = _gateway(
        engine_session=EngineSession(state),
        discovered_memory=DiscoveredClinicalMemory(),
    )

    observation = gateway.for_clinician()

    assert 222 not in _values_in(observation)
    assert 66 not in _values_in(observation)


def test_clinician_discovered_allergy_overrides_ground_truth_allergy():
    discovered = DiscoveredClinicalMemory()
    discovered.discover_allergies(["penicillin"])
    gateway = _gateway(
        ground_truth_memory=GroundTruthMemory(allergies=["aspirin"]),
        discovered_memory=discovered,
    )

    observation = gateway.for_clinician()

    assert observation.allergies == ["penicillin"]
    assert "aspirin" not in _values_in(observation)


def test_for_role_dispatches_case_insensitive_role_names():
    gateway = _gateway()

    assert isinstance(gateway.for_role("clinician"), ClinicianObservation)
    assert isinstance(gateway.for_role("NURSE"), NurseObservation)
    with pytest.raises(ValueError, match="unsupported observation role"):
        gateway.for_role("physician")


def test_nurse_sees_current_observable_vitals():
    session = EngineSession(_base_state())

    observation = _gateway(engine_session=session).for_nurse()

    assert observation.current_vitals == session.observe_vitals()


def test_nurse_sees_pending_orders_active_task_and_queue_summary():
    workflow = WorkflowEngine()
    first_order = workflow.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 30},
    )
    second_order = workflow.submit_order(
        order_type="lab",
        payload={"test_name": "glucose", "duration_s": 30},
    )
    workflow.advance(1)

    observation = _gateway(workflow_engine=workflow).for_nurse()

    assert {order["order_id"] for order in observation.pending_orders} == {
        first_order.order_id,
        second_order.order_id,
    }
    assert observation.active_task is not None
    assert observation.active_task["source_order_id"] == first_order.order_id
    assert observation.task_queue_summary["active_count"] == 1
    assert observation.task_queue_summary["queued_count"] == 1
    assert observation.queued_tasks[0]["source_order_id"] == second_order.order_id


def test_patient_observation_contains_subjective_state_but_not_numeric_vitals():
    observation = _gateway(subjective_state=_subjective_state()).for_patient()

    assert observation.subjective_state == _subjective_state()
    assert _keys_in(observation).isdisjoint(NUMERIC_VITAL_KEYS)
    assert _keys_in(observation).isdisjoint(FORBIDDEN_OBSERVATION_KEYS)


def test_patient_subjective_state_falls_back_to_observable_vitals():
    patient = _gateway().for_patient()

    assert isinstance(patient.subjective_state, SubjectiveState)
    assert patient.subjective_state.dyspnea_severity > 0


def test_patient_observation_includes_private_symptom_timeline():
    patient_private = PatientPrivateMemory(
        symptom_knowledge={"location": "chest", "HR": 120},
        symptom_timeline={"onset": "45 minutes ago", "progression": "sudden"},
    )

    observation = _gateway(
        patient_private_memory=patient_private,
        subjective_state=_subjective_state(),
    ).for_patient()

    assert observation.patient_private_memory["symptom_timeline"] == {
        "onset": "45 minutes ago",
        "progression": "sudden",
    }
    assert observation.patient_private_memory["symptom_knowledge"] == {
        "location": "chest"
    }
    assert "HR" not in _keys_in(observation)


def test_relative_observation_contains_visible_state_and_private_memory_only():
    relative_private = RelativePrivateMemory(
        relationship="spouse",
        private_facts={"called_ems": True, "O2Sat": 88},
    )
    conversation = ConversationMemory(
        emotional_or_social_summary="Spouse is worried and asking for updates."
    )
    conversation.add_turn(
        speaker="nurse",
        target="relative",
        content="We are watching them closely.",
        metadata={"BP_sys": 92},
    )

    observation = _gateway(
        relative_private_memory=relative_private,
        conversation_memory=conversation,
        subjective_state=_subjective_state(),
    ).for_relative()

    assert observation.visible_patient_state["visible_distress"] == 0.75
    assert observation.visible_patient_state["speech_capacity"] == "short_phrases"
    assert observation.relative_private_memory["relationship"] == "spouse"
    assert observation.relative_private_memory["private_facts"] == {
        "called_ems": True
    }
    assert observation.emotional_or_social_context == (
        "Spouse is worried and asking for updates."
    )
    assert _keys_in(observation).isdisjoint(NUMERIC_VITAL_KEYS)
    assert _keys_in(observation).isdisjoint(FORBIDDEN_OBSERVATION_KEYS)


def test_patient_and_relative_response_opportunity_metadata_strips_numeric_vitals():
    queue = ResponseOpportunityQueue()
    queue.add(
        ResponseOpportunity(
            opportunity_id="patient-pain",
            addressed_to="patient",
            question_text="Where is your pain?",
            metadata={"HR": 120, "topic": "pain"},
        )
    )
    queue.add(
        ResponseOpportunity(
            opportunity_id="relative-history",
            addressed_to="relative",
            question_text="What happened before arrival?",
            metadata={"O2Sat": 88, "topic": "history"},
        )
    )
    gateway = _gateway(
        response_opportunities=queue,
        subjective_state=_subjective_state(),
    )

    patient = gateway.for_patient()
    relative = gateway.for_relative()

    assert patient.response_opportunities[0]["metadata"] == {"topic": "pain"}
    assert relative.response_opportunities[0]["metadata"] == {"topic": "history"}
    assert "HR" not in _keys_in(patient)
    assert "O2Sat" not in _keys_in(relative)


def test_clinician_retains_discovered_numeric_vitals_history():
    discovered = DiscoveredClinicalMemory(
        vitals_history=[{"HR": 120, "O2Sat": 92}],
    )

    observation = _gateway(discovered_memory=discovered).for_clinician()

    assert observation.discovered_vitals_history == [{"HR": 120, "O2Sat": 92}]


def test_clinician_observation_mutation_does_not_mutate_discovered_memory():
    discovered = DiscoveredClinicalMemory()
    discovered.discover_allergies(["penicillin"])

    observation = _gateway(discovered_memory=discovered).for_clinician()
    observation.allergies.append("MUTATED")

    assert discovered.allergies == ["penicillin"]


def test_no_role_observation_contains_forbidden_keys_at_any_depth():
    discovered = DiscoveredClinicalMemory(
        symptom_history={"mechanism": "leak", "onset": "today"},
        vitals_history=[{"HR": 90, "pathology": "leak"}],
        exam_findings={"active_drug_effects": ["leak"], "lungs": "wheeze"},
        test_results={"ecg": {"_hidden_state": "leak", "finding": "SVT"}},
    )
    conversation = ConversationMemory()
    conversation.add_turn(
        speaker="clinician",
        target="patient",
        content="How do you feel?",
        metadata={"CO_index": 0.4},
    )
    patient_private = PatientPrivateMemory(
        symptom_knowledge={"SVR_index": 0.5, "complaint": "shortness of breath"}
    )
    relative_private = RelativePrivateMemory(
        private_facts={"PaO2_effective": 55, "arrival": "called EMS"}
    )
    workflow = WorkflowEngine()
    workflow.submit_order(
        order_type="intervention",
        payload={
            "name": "apply_NRB",
            "duration_s": 30,
            "emsim_action": {"mechanism": "leak"},
        },
    )
    workflow.advance(1)

    gateway = _gateway(
        workflow_engine=workflow,
        discovered_memory=discovered,
        conversation_memory=conversation,
        patient_private_memory=patient_private,
        relative_private_memory=relative_private,
        subjective_state=_subjective_state(),
    )

    observations = [
        gateway.for_clinician(),
        gateway.for_nurse(),
        gateway.for_patient(),
        gateway.for_relative(),
    ]

    for observation in observations:
        assert _keys_in(observation).isdisjoint(FORBIDDEN_OBSERVATION_KEYS)


def test_observation_gateway_does_not_import_rule_engine():
    source = (EMSIM_ROOT / "ed_multiagent/observation/gateway.py").read_text(
        encoding="utf-8"
    )

    assert "rule_engine" not in source
