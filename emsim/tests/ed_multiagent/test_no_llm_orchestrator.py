from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.actions import ResponseMode
from ed_multiagent.emsim_adapter import EngineSession
from ed_multiagent.memory import (
    ConversationMemory,
    DiscoveredClinicalMemory,
    GroundTruthMemory,
    PatientPrivateMemory,
    RelativePrivateMemory,
)
from ed_multiagent.observation import ObservationGateway
from ed_multiagent.orchestrator import (
    DECLARE_CODE,
    EXIT_CODE,
    ClinicianTurn,
    SimMode,
    SimModeManager,
    SimulationOrchestrator,
    VerbalAction,
)
from ed_multiagent.world import PendingEvent, WorkflowEngine


FORBIDDEN_CLINICIAN_KEYS = {
    "hidden_state",
    "mechanism",
    "pathology",
    "rhythm",
    "interventions",
    "active_drug_effects",
}


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


def _orchestrator(**kwargs: Any) -> SimulationOrchestrator:
    kwargs.setdefault("engine_session", EngineSession(_base_state()))
    kwargs.setdefault("workflow_engine", WorkflowEngine())
    return SimulationOrchestrator(**kwargs)


def _keys_in(value: Any) -> set[str]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
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


def test_constructs_from_existing_components():
    session = EngineSession(_base_state())
    workflow = WorkflowEngine()
    conversation = ConversationMemory()
    discovered = DiscoveredClinicalMemory()
    gateway = ObservationGateway(
        engine_session=session,
        workflow_engine=workflow,
        ground_truth_memory=GroundTruthMemory(allergies=["penicillin"]),
        discovered_memory=discovered,
        conversation_memory=conversation,
        patient_private_memory=PatientPrivateMemory(),
        relative_private_memory=RelativePrivateMemory(),
    )
    sim_mode_manager = SimModeManager()

    orchestrator = SimulationOrchestrator(
        session,
        workflow,
        conversation_memory=conversation,
        discovered_memory=discovered,
        observation_gateway=gateway,
        sim_mode_manager=sim_mode_manager,
    )

    assert orchestrator.engine_session is session
    assert orchestrator.workflow_engine is workflow
    assert orchestrator.conversation_memory is conversation
    assert orchestrator.discovered_memory is discovered
    assert orchestrator.observation_gateway is gateway
    assert orchestrator.sim_mode_manager is sim_mode_manager


def test_structured_intervention_order_flows_through_workflow_to_engine():
    orchestrator = _orchestrator()
    before_o2 = orchestrator.engine_session.observe_vitals()["O2Sat"]

    result = orchestrator.run_round(
        ClinicianTurn(
            order_requests=[
                {
                    "order_type": "intervention",
                    "payload": {"name": "apply_NRB", "duration_s": 30},
                }
            ],
            wait_s=30,
        )
    )

    assert result.submitted_orders[0].status == "completed"
    assert [task.task_type for task in result.workflow_result.completed_tasks] == [
        "apply_intervention"
    ]
    assert result.applied_emsim_actions == [
        {"type": "intervention", "name": "apply_NRB"}
    ]
    assert orchestrator.round_logger.entries[-1].applied_emsim_actions == [
        {"type": "intervention", "name": "apply_NRB"}
    ]
    assert orchestrator.engine_session.observe_vitals()["O2Sat"] > before_o2
    orchestrator.engine_session.advance(1)


def test_structured_drug_order_flows_through_workflow_to_engine():
    orchestrator = _orchestrator()
    before_bp = orchestrator.engine_session.observe_vitals()["BP_sys"]

    result = orchestrator.run_round(
        ClinicianTurn(
            order_requests=[
                {
                    "order_type": "drug",
                    "payload": {
                        "name": "epinephrine",
                        "dose": 1,
                        "unit": "mg",
                        "route": "IV",
                        "duration_s": 30,
                    },
                }
            ],
            wait_s=30,
        )
    )

    assert result.applied_emsim_actions == [
        {
            "type": "drug",
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
        }
    ]
    assert orchestrator.engine_session.observe_vitals()["BP_sys"] > before_bp
    orchestrator.engine_session.advance(1)


def test_lab_result_ready_event_updates_discovered_memory_after_turnaround():
    discovered = DiscoveredClinicalMemory()
    conversation = ConversationMemory()
    orchestrator = _orchestrator(
        discovered_memory=discovered,
        conversation_memory=conversation,
    )
    turn = ClinicianTurn(
        order_requests=[
            {
                "order_type": "lab",
                "payload": {
                    "test_name": "glucose",
                    "result": {"value": 180, "unit": "mg/dL"},
                    "duration_s": 90,
                    "turnaround_s": 30,
                },
            }
        ],
        wait_s=90,
    )

    draw_round = orchestrator.run_round(turn)
    almost_ready = orchestrator.run_round(ClinicianTurn(wait_s=29))
    ready = orchestrator.run_round(ClinicianTurn(wait_s=1))

    assert draw_round.fired_events == []
    assert almost_ready.fired_events == []
    assert [event.event_type for event in ready.fired_events] == ["lab_result_ready"]
    assert discovered.test_results == {
        "glucose": {"value": 180, "unit": "mg/dL"}
    }
    assert draw_round.submitted_orders[0].status == "completed"
    assert conversation.raw_turns == []


@pytest.mark.parametrize("source_order_id", [None, "purged-order"])
def test_lab_result_event_without_resolvable_order_still_updates_memory(
    source_order_id: str | None,
):
    discovered = DiscoveredClinicalMemory()
    orchestrator = _orchestrator(discovered_memory=discovered)
    memory_deltas = {"test_results_added": {}}
    event = PendingEvent(
        event_type="lab_result_ready",
        payload={
            "test_name": "glucose",
            "result": {"value": 180, "unit": "mg/dL"},
        },
        source_order_id=source_order_id,
    )

    orchestrator._process_fired_events([event], memory_deltas)

    assert discovered.test_results == {
        "glucose": {"value": 180, "unit": "mg/dL"}
    }
    assert memory_deltas["test_results_added"] == {
        "glucose": {"value": 180, "unit": "mg/dL"}
    }


def test_lab_result_event_processing_does_not_complete_order_status(monkeypatch):
    orchestrator = _orchestrator()
    order = orchestrator.workflow_engine.submit_order(
        order_type="lab",
        payload={"test_name": "glucose"},
    )
    original_set_status = orchestrator.workflow_engine.order_manager.set_status
    status_updates = []

    def spy_set_status(
        order_id: str,
        status: str,
        *,
        now_s: float | None = None,
    ):
        status_updates.append((order_id, status))
        return original_set_status(order_id, status, now_s=now_s)

    monkeypatch.setattr(
        orchestrator.workflow_engine.order_manager,
        "set_status",
        spy_set_status,
    )
    event = PendingEvent(
        event_type="lab_result_ready",
        payload={"test_name": "glucose"},
        source_order_id=order.order_id,
    )

    orchestrator._process_fired_events([event], {"test_results_added": {}})

    assert status_updates == []
    assert order.status == "pending_execution"


def test_question_without_target_or_addressed_to_raises():
    with pytest.raises(ValueError, match="target or addressed_to"):
        VerbalAction(content="Any allergies?", is_question=True)


def test_response_opportunity_hardcoded_answers_update_only_answered_facts():
    discovered = DiscoveredClinicalMemory()
    orchestrator = _orchestrator(discovered_memory=discovered)

    result = orchestrator.run_round(
        ClinicianTurn(
            verbal_action=VerbalAction(
                target="patient",
                content="Any allergies?",
                is_question=True,
                expects_response=True,
                question_type="allergy",
                expected_slots=["allergies"],
            ),
            wait_s=0,
        )
    )
    opportunity = result.response_opportunities_created[0]

    resolved = orchestrator.apply_hardcoded_response(
        opportunity.opportunity_id,
        ResponseMode.DIRECT_ANSWER,
        {"allergies": ["penicillin"]},
    )

    assert resolved.status.value == "answered"
    assert discovered.allergies_known is True
    assert discovered.allergies == ["penicillin"]

    refused_round = orchestrator.run_round(
        ClinicianTurn(
            verbal_action=VerbalAction(
                target="patient",
                content="What medications do you take?",
                expects_response=True,
                question_type="medication",
                expected_slots=["medications"],
            ),
            wait_s=0,
        )
    )
    refused = refused_round.response_opportunities_created[0]
    orchestrator.apply_hardcoded_response(
        refused.opportunity_id,
        ResponseMode.REFUSE,
        {"medications": ["metformin"]},
    )

    assert discovered.medications_known is False
    assert discovered.medications == []


@pytest.mark.parametrize(
    "response_mode",
    [ResponseMode.EVADE, ResponseMode.UNABLE],
)
def test_evaded_or_unable_response_does_not_update_clinical_facts(
    response_mode: ResponseMode,
):
    discovered = DiscoveredClinicalMemory()
    orchestrator = _orchestrator(discovered_memory=discovered)
    result = orchestrator.run_round(
        ClinicianTurn(
            verbal_action={
                "target": "patient",
                "content": "Any daily medications?",
                "expects_response": True,
                "expected_slots": ["medications"],
            },
            wait_s=0,
        )
    )

    orchestrator.apply_hardcoded_response(
        result.response_opportunities_created[0].opportunity_id,
        response_mode,
        {"medications": ["metformin"]},
    )

    assert discovered.medications_known is False
    assert discovered.medications == []


def test_clinician_observation_uses_gateway_and_not_engine_debug_state():
    orchestrator = _orchestrator()

    result = orchestrator.run_round(ClinicianTurn(wait_s=0))
    clinician_observation = result.observations["clinician"]

    assert _keys_in(clinician_observation).isdisjoint(FORBIDDEN_CLINICIAN_KEYS)
    assert clinician_observation is not result.engine_result.state
    assert getattr(clinician_observation, "available_test_results") == {}


def test_sim_mode_code_meta_actions_and_no_llm_agent_calls():
    orchestrator = _orchestrator(sim_mode_manager=SimModeManager())

    stable = orchestrator.run_round(ClinicianTurn())
    declared = orchestrator.run_round(ClinicianTurn(meta_action=DECLARE_CODE))
    exit_attempt = orchestrator.run_round(ClinicianTurn(meta_action=EXIT_CODE))

    assert stable.sim_mode == SimMode.STABLE
    assert stable.dt_s == pytest.approx(30.0)
    assert declared.sim_mode == SimMode.CODE
    assert declared.dt_s == pytest.approx(5.0)
    assert exit_attempt.sim_mode == SimMode.CODE
    assert exit_attempt.next_sim_mode == SimMode.CODE
    assert exit_attempt.dt_s == pytest.approx(5.0)
    assert declared.agent_calls == []
    assert exit_attempt.agent_calls == []
    assert exit_attempt.activation_recommendations["clinician_agent"] is False


def test_sim_mode_hysteresis_updates_once_per_run_round():
    manager = SimModeManager(
        downgrade_required_rounds=3,
        initial_mode=SimMode.URGENT,
    )
    orchestrator = _orchestrator(sim_mode_manager=manager)

    first = orchestrator.run_round(ClinicianTurn(wait_s=0))
    second = orchestrator.run_round(ClinicianTurn(wait_s=0))
    third = orchestrator.run_round(ClinicianTurn(wait_s=0))

    assert first.sim_mode == SimMode.URGENT
    assert second.sim_mode == SimMode.URGENT
    assert third.sim_mode == SimMode.STABLE


def test_resolved_response_opportunities_are_returned_and_logged():
    orchestrator = _orchestrator()
    question_round = orchestrator.run_round(
        ClinicianTurn(
            verbal_action=VerbalAction(
                target="patient",
                content="Any allergies?",
                expects_response=True,
                question_type="allergy",
                expected_slots=["allergies"],
            ),
            wait_s=0,
        )
    )
    opportunity = question_round.response_opportunities_created[0]

    orchestrator.apply_hardcoded_response(
        opportunity.opportunity_id,
        ResponseMode.DIRECT_ANSWER,
        {"allergies": ["penicillin"]},
    )
    replay_round = orchestrator.run_round(ClinicianTurn(wait_s=0))

    assert [
        resolved.opportunity_id
        for resolved in replay_round.response_opportunities_resolved
    ] == [opportunity.opportunity_id]
    assert replay_round.response_opportunities_resolved[0].status.value == "answered"
    assert orchestrator.round_logger.entries[-1].response_opportunities_resolved == [
        {
            "opportunity_id": opportunity.opportunity_id,
            "asked_by": "clinician",
            "addressed_to": "patient",
            "question_text": "Any allergies?",
            "question_type": "allergy",
            "requiredness": "expected",
            "sensitivity": "low",
            "urgency": "routine",
            "expected_slots": ["allergies"],
            "created_at_s": 0.0,
            "status": "answered",
            "metadata": {},
        }
    ]


def test_workflow_clock_mismatch_raises_instead_of_silent_rewind():
    session = EngineSession(_base_state())
    workflow = WorkflowEngine(now_s=30)

    with pytest.raises(ValueError, match="workflow_engine.now_s must match"):
        SimulationOrchestrator(session, workflow)


def test_no_llm_orchestrator_does_not_import_rule_engine_internals():
    source = (EMSIM_ROOT / "ed_multiagent/orchestrator/simulation.py").read_text(
        encoding="utf-8"
    )

    assert "rule_engine" not in source
    assert "drug_lib" not in source
    assert "pathology_lib" not in source
    assert "intervention_lib" not in source
    assert "_hidden_state" not in source
