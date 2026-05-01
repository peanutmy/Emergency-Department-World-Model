from __future__ import annotations

from pathlib import Path
import sys

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.emsim_adapter import EngineSession
from ed_multiagent.world import (
    ClinicalOrder,
    NurseTask,
    NurseTaskQueue,
    OrderManager,
    PendingEventQueue,
    WorkflowEngine,
)


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


def test_drug_order_creates_a_nurse_task():
    engine = WorkflowEngine()

    order = engine.submit_order(
        order_type="drug",
        payload={
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
        },
    )

    tasks = engine.nurse_task_queue.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].source_order_id == order.order_id
    assert tasks[0].task_type == "administer_drug"
    assert tasks[0].emsim_action == {
        "type": "drug",
        "name": "epinephrine",
        "dose": 1,
        "unit": "mg",
        "route": "IV",
    }


def test_intervention_order_creates_a_nurse_task_with_emsim_action():
    engine = WorkflowEngine()

    engine.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB"},
    )

    task = engine.nurse_task_queue.list_tasks()[0]
    assert task.task_type == "apply_intervention"
    assert task.duration_s == pytest.approx(30.0)
    assert task.emsim_action == {
        "type": "intervention",
        "name": "apply_NRB",
    }


def test_lab_order_creates_draw_task_and_later_lab_result_ready_event():
    engine = WorkflowEngine()

    order = engine.submit_order(
        order_type="lab",
        payload={
            "test_name": "glucose",
            "result": {"value": 180, "unit": "mg/dL"},
        },
    )
    task = engine.nurse_task_queue.list_tasks()[0]
    assert task.task_type == "draw_lab"

    draw_result = engine.advance(90)
    assert [completed.task_type for completed in draw_result.completed_tasks] == [
        "draw_lab"
    ]
    assert draw_result.events == []
    assert engine.event_queue.pending_events()[0].event_type == "lab_result_ready"

    assert engine.advance(29).events == []
    ready = engine.advance(1).events

    assert len(ready) == 1
    assert ready[0].event_type == "lab_result_ready"
    assert ready[0].source_order_id == order.order_id
    assert ready[0].payload == {
        "test_name": "glucose",
        "result": {"value": 180, "unit": "mg/dL"},
    }
    assert engine.order_manager.get_order(order.order_id).status == "completed"


def test_remaining_s_decreases_by_dt_s():
    engine = WorkflowEngine()
    engine.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 45},
    )

    engine.advance(10)

    task = engine.nurse_task_queue.list_tasks()[0]
    assert task.remaining_s == pytest.approx(35.0)


def test_task_duration_remains_correct_across_dt_s_changes():
    engine = WorkflowEngine()
    engine.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 40},
    )
    task = engine.nurse_task_queue.list_tasks()[0]

    engine.advance(30)
    assert task.duration_s == pytest.approx(40.0)
    assert task.remaining_s == pytest.approx(10.0)

    engine.advance(5)
    assert task.duration_s == pytest.approx(40.0)
    assert task.remaining_s == pytest.approx(5.0)

    result = engine.advance(5)
    assert task.duration_s == pytest.approx(40.0)
    assert task.remaining_s == pytest.approx(0.0)
    assert [completed.task_id for completed in result.completed_tasks] == [
        task.task_id
    ]


def test_completed_task_returns_emsim_action_that_engine_session_accepts():
    workflow = WorkflowEngine()
    workflow.submit_order(
        order_type="drug",
        payload={
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
            "duration_s": 30,
        },
    )
    session = EngineSession(_base_state())

    result = workflow.advance(30)

    assert len(result.emsim_actions) == 1
    session.apply_emsim_action(result.emsim_actions[0])
    first = session.advance(30).state["vitals"]
    second = session.advance(30).state["vitals"]

    assert second["BP_sys"] > first["BP_sys"]


def test_completed_task_complete_returns_a_defensive_action_copy():
    task = NurseTask(
        task_id="task-1",
        task_type="apply_intervention",
        duration_s=5,
        emsim_action={"type": "intervention", "name": "apply_NRB"},
    )
    task.start(now_s=0)
    assert task.advance(5, now_s=5)

    action = task.complete(now_s=5)

    assert action == {"type": "intervention", "name": "apply_NRB"}
    action["name"] = "mutated"
    assert task.emsim_action == {"type": "intervention", "name": "apply_NRB"}


def test_fifo_activation_ignores_priority_for_m1b():
    queue = NurseTaskQueue()
    routine = queue.create_task(
        task_type="apply_intervention",
        priority="routine",
        duration_s=30,
    )
    stat = queue.create_task(
        task_type="apply_intervention",
        priority="stat",
        duration_s=30,
    )

    activated = queue.activate_queued(now_s=0)

    assert [task.task_id for task in activated] == [routine.task_id]
    assert routine.status == "active"
    assert stat.status == "queued"


def test_multi_task_queueing_under_capacity_one():
    engine = WorkflowEngine()
    first_order = engine.submit_order(
        order_type="intervention",
        priority="routine",
        payload={"name": "apply_NRB", "duration_s": 10},
    )
    second_order = engine.submit_order(
        order_type="drug",
        priority="stat",
        payload={
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
            "duration_s": 10,
        },
    )
    first_task, second_task = engine.nurse_task_queue.list_tasks()

    engine.advance(1)

    assert first_task.status == "active"
    assert second_task.status == "queued"
    assert engine.order_manager.get_order(first_order.order_id).status == "in_progress"
    assert engine.order_manager.get_order(second_order.order_id).status == (
        "pending_execution"
    )

    engine.advance(9)

    assert first_task.status == "done"
    assert second_task.status == "active"
    assert engine.order_manager.get_order(second_order.order_id).status == "in_progress"


def test_workflow_advance_negative_raises_value_error():
    engine = WorkflowEngine()

    with pytest.raises(ValueError):
        engine.advance(-1)


def test_unsupported_order_type_is_rejected():
    engine = WorkflowEngine()

    with pytest.raises(ValueError, match="unsupported order_type"):
        engine.submit_order(order_type="imaging", payload={"name": "cxr"})


def test_drug_and_intervention_completion_do_not_schedule_lab_events():
    engine = WorkflowEngine()
    engine.submit_order(
        order_type="drug",
        payload={
            "name": "epinephrine",
            "dose": 1,
            "unit": "mg",
            "route": "IV",
            "duration_s": 10,
        },
    )
    engine.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 10},
    )

    first = engine.advance(10)
    second = engine.advance(10)

    assert len(first.emsim_actions) == 1
    assert len(second.emsim_actions) == 1
    assert first.events == []
    assert second.events == []
    assert engine.event_queue.pending_events() == []


def test_order_status_progression():
    engine = WorkflowEngine()
    order = ClinicalOrder(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 10},
    )
    statuses = [order.status]

    engine.submit_order(order)
    statuses.append(order.status)
    engine.advance(5)
    statuses.append(order.status)
    engine.advance(5)
    statuses.append(order.status)

    assert statuses == [
        "ordered",
        "pending_execution",
        "in_progress",
        "completed",
    ]


def test_pending_event_queue_schedule_event_with_delay_s():
    queue = PendingEventQueue(now_s=10)

    event = queue.schedule_event(
        event_type="lab_result_ready",
        delay_s=5,
        payload={"test_name": "glucose"},
    )

    assert event.due_at_s == pytest.approx(15.0)
    assert queue.advance(4) == []
    assert queue.advance(1) == [event]


def test_pending_event_queue_schedule_event_with_due_at_s():
    queue = PendingEventQueue(now_s=10)

    event = queue.schedule_event(
        event_type="lab_result_ready",
        due_at_s=12,
        payload={"test_name": "glucose"},
    )

    assert event.due_at_s == pytest.approx(12.0)
    assert queue.advance(2) == [event]


def test_pending_event_queue_schedule_event_requires_delay_or_due_time():
    queue = PendingEventQueue()

    with pytest.raises(ValueError, match="due_at_s or delay_s is required"):
        queue.schedule_event(event_type="lab_result_ready")


def test_delivered_event_remains_accessible_for_audit_replay():
    queue = PendingEventQueue(now_s=0)
    event = queue.schedule_event(
        event_type="lab_result_ready",
        delay_s=5,
        payload={"test_name": "glucose"},
    )

    delivered = queue.advance(5)

    assert delivered == [event]
    assert queue.get_event(event.event_id) is event
    assert queue.get_event(event.event_id).status == "delivered"
    assert queue.pending_events() == []


def test_lab_event_payload_shape_when_result_is_none():
    engine = WorkflowEngine()

    engine.submit_order(
        order_type="lab",
        payload={"test_name": "glucose"},
    )
    engine.advance(90)
    ready = engine.advance(30).events

    assert len(ready) == 1
    assert ready[0].payload == {"test_name": "glucose"}


def test_order_and_task_payloads_are_defensive_copies_at_submit_time():
    engine = WorkflowEngine()
    payload = {
        "name": "apply_NRB",
        "duration_s": 30,
        "nested": {"original": True},
    }

    order = engine.submit_order(order_type="intervention", payload=payload)
    task = engine.nurse_task_queue.list_tasks()[0]
    payload["name"] = "mutated_original"
    payload["nested"]["original"] = False
    order.payload["name"] = "mutated_order"
    task.payload["name"] = "mutated_task"

    assert order.payload["nested"] == {"original": True}
    assert task.payload["nested"] == {"original": True}
    assert task.emsim_action == {
        "type": "intervention",
        "name": "apply_NRB",
        "nested": {"original": True},
    }


def test_workflow_advance_zero_is_no_op_and_keeps_event_queue_clock():
    engine = WorkflowEngine(now_s=10)
    engine.event_queue.now_s = 10
    event = engine.event_queue.schedule_event(
        event_type="lab_result_ready",
        due_at_s=10,
        payload={"test_name": "glucose"},
    )
    engine.submit_order(
        order_type="intervention",
        payload={"name": "apply_NRB", "duration_s": 30},
    )
    task = engine.nurse_task_queue.list_tasks()[0]

    result = engine.advance(0)

    assert result.now_s == pytest.approx(10.0)
    assert result.completed_tasks == []
    assert result.emsim_actions == []
    assert result.events == []
    assert engine.event_queue.now_s == pytest.approx(10.0)
    assert event.status == "pending"
    assert task.status == "queued"


def test_direct_queued_task_advance_does_not_auto_start():
    task = NurseTask(task_id="task-1", task_type="apply_intervention", duration_s=10)

    completed = task.advance(5, now_s=5)

    assert completed is False
    assert task.status == "queued"
    assert task.remaining_s == pytest.approx(10.0)
    assert task.started_at_s is None


def test_order_manager_rejects_incoming_orders_not_in_ordered_status():
    manager = OrderManager()
    order = ClinicalOrder(
        order_type="drug",
        payload={"name": "epinephrine"},
        status="in_progress",
    )

    with pytest.raises(ValueError, match="status 'ordered'"):
        manager.add_order(order)


def test_order_manager_submit_order_requires_order_type_not_type_alias():
    manager = OrderManager()

    with pytest.raises(ValueError, match="order_type is required"):
        manager.submit_order({"type": "drug", "payload": {"name": "epinephrine"}})
