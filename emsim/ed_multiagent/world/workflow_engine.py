"""Deterministic M1b workflow engine for structured orders."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .events import PendingEvent, PendingEventQueue
from .nurse_tasks import (
    DEFAULT_TASK_DURATION_S,
    INTERVENTION_TASK_DURATION_S,
    NurseTask,
    NurseTaskQueue,
)
from .orders import ClinicalOrder, OrderManager


LAB_TURNAROUND_S: dict[str, float] = {
    "glucose": 30.0,
    "ecg": 60.0,
    "abg": 180.0,
    "vbg": 180.0,
    "lactate": 360.0,
    "cbc": 600.0,
    "bmp": 600.0,
    "troponin": 900.0,
    "cxr": 600.0,
    "ct_head": 1800.0,
}

_WORKFLOW_PAYLOAD_KEYS = {
    "duration_s",
    "task_duration_s",
    "turnaround_s",
    "result",
    "result_payload",
    "emsim_action",
}


@dataclass
class WorkflowAdvanceResult:
    """Deterministic workflow outputs from one simulated time advance."""

    now_s: float
    completed_tasks: list[NurseTask] = field(default_factory=list)
    emsim_actions: list[dict[str, Any]] = field(default_factory=list)
    events: list[PendingEvent] = field(default_factory=list)

    @property
    def ready_events(self) -> list[PendingEvent]:
        return self.events

    @property
    def emsim_action(self) -> dict[str, Any] | None:
        if not self.emsim_actions:
            return None
        return self.emsim_actions[0]

    @property
    def completed_task(self) -> NurseTask | None:
        if not self.completed_tasks:
            return None
        return self.completed_tasks[0]


class WorkflowEngine:
    """Route hand-typed orders through delayed nurse tasks and events."""

    def __init__(
        self,
        *,
        now_s: float = 0.0,
        order_manager: OrderManager | None = None,
        nurse_task_queue: NurseTaskQueue | None = None,
        event_queue: PendingEventQueue | None = None,
        lab_turnaround_s: dict[str, float] | None = None,
    ):
        self.now_s = float(now_s)
        self.order_manager = order_manager or OrderManager()
        self.nurse_task_queue = nurse_task_queue or NurseTaskQueue()
        self.task_queue = self.nurse_task_queue
        self.event_queue = event_queue or PendingEventQueue()
        self.lab_turnaround_s = dict(LAB_TURNAROUND_S)
        if lab_turnaround_s:
            self.lab_turnaround_s.update(
                {name: float(delay_s) for name, delay_s in lab_turnaround_s.items()}
            )

    @property
    def orders(self) -> dict[str, ClinicalOrder]:
        return self.order_manager.orders

    @property
    def nurse_tasks(self) -> dict[str, NurseTask]:
        return self.nurse_task_queue.tasks

    @property
    def pending_events(self) -> list[PendingEvent]:
        return self.event_queue.pending_events()

    def submit_order(
        self,
        order: ClinicalOrder | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClinicalOrder:
        """Store an order and create its first workflow task."""

        order_kwargs = dict(order or {}) if isinstance(order, dict) else {}
        order_kwargs.update(kwargs)
        if "ordered_at_s" not in order_kwargs:
            order_kwargs["ordered_at_s"] = self.now_s

        clinical_order = self.order_manager.submit_order(
            order if isinstance(order, ClinicalOrder) else order_kwargs
        )
        self._create_workflow_for_order(clinical_order)
        return clinical_order

    def advance(self, dt_s: float) -> WorkflowAdvanceResult:
        """Advance workflow tasks and pending events by simulated seconds."""

        dt_s = float(dt_s)
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        if dt_s == 0:
            return WorkflowAdvanceResult(now_s=self.now_s)

        activated_tasks = self.nurse_task_queue.activate_queued(now_s=self.now_s)
        for task in activated_tasks:
            if task.source_order_id is not None:
                self.order_manager.set_status(
                    task.source_order_id,
                    "in_progress",
                    now_s=self.now_s,
                )

        self.now_s += dt_s
        completed_tasks = self.nurse_task_queue.advance_active(
            dt_s,
            now_s=self.now_s,
        )

        emsim_actions: list[dict[str, Any]] = []
        for task in completed_tasks:
            emsim_action = task.complete(now_s=self.now_s)
            if emsim_action is not None:
                emsim_actions.append(emsim_action)
                if task.source_order_id is not None:
                    self.order_manager.set_status(
                        task.source_order_id,
                        "completed",
                        now_s=self.now_s,
                    )
            elif task.task_type == "draw_lab":
                self._schedule_lab_result(task)

        self._activate_next_queued_tasks()

        # PendingEventQueue also supports standalone relative scheduling. Keep
        # its clock aligned here because WorkflowEngine owns simulated time.
        self.event_queue.now_s = self.now_s
        events = self.event_queue.pop_due(self.now_s)
        for event in events:
            if (
                event.event_type == "lab_result_ready"
                and event.source_order_id is not None
            ):
                self.order_manager.set_status(
                    event.source_order_id,
                    "completed",
                    now_s=self.now_s,
                )

        return WorkflowAdvanceResult(
            now_s=self.now_s,
            completed_tasks=completed_tasks,
            emsim_actions=emsim_actions,
            events=events,
        )

    def _create_workflow_for_order(self, order: ClinicalOrder) -> NurseTask:
        if order.order_type == "drug":
            task = self._create_drug_task(order)
        elif order.order_type == "intervention":
            task = self._create_intervention_task(order)
        elif order.order_type == "lab":
            task = self._create_lab_task(order)
        else:
            raise ValueError(f"unsupported order_type {order.order_type!r}")

        self.order_manager.set_status(
            order.order_id,
            "pending_execution",
            now_s=self.now_s,
        )
        return task

    def _create_drug_task(self, order: ClinicalOrder) -> NurseTask:
        payload = deepcopy(order.payload)
        # Snapshot the EMSim action at submit time; later payload edits do not
        # re-derive the action that the nurse task will execute.
        action = self._emsim_action("drug", payload)
        return self.nurse_task_queue.create_task(
            task_type="administer_drug",
            source_order_id=order.order_id,
            priority=order.priority,
            duration_s=self._duration_from_payload(
                payload,
                DEFAULT_TASK_DURATION_S["administer_drug"],
            ),
            payload=payload,
            emsim_action=action,
            created_at_s=self.now_s,
        )

    def _create_intervention_task(self, order: ClinicalOrder) -> NurseTask:
        payload = deepcopy(order.payload)
        name = str(payload.get("name", ""))
        default_duration_s = INTERVENTION_TASK_DURATION_S.get(
            name,
            DEFAULT_TASK_DURATION_S["apply_intervention"],
        )
        # Snapshot the EMSim action at submit time; later payload edits do not
        # re-derive the action that the nurse task will execute.
        action = self._emsim_action("intervention", payload)
        return self.nurse_task_queue.create_task(
            task_type="apply_intervention",
            source_order_id=order.order_id,
            priority=order.priority,
            duration_s=self._duration_from_payload(payload, default_duration_s),
            payload=payload,
            emsim_action=action,
            created_at_s=self.now_s,
        )

    def _activate_next_queued_tasks(self) -> None:
        activated_tasks = self.nurse_task_queue.activate_queued(now_s=self.now_s)
        for task in activated_tasks:
            if task.source_order_id is not None:
                self.order_manager.set_status(
                    task.source_order_id,
                    "in_progress",
                    now_s=self.now_s,
                )

    def _create_lab_task(self, order: ClinicalOrder) -> NurseTask:
        payload = deepcopy(order.payload)
        test_name = self._lab_name(payload)
        payload["test_name"] = test_name
        payload["turnaround_s"] = self._lab_turnaround_s(payload)
        return self.nurse_task_queue.create_task(
            task_type="draw_lab",
            source_order_id=order.order_id,
            priority=order.priority,
            duration_s=self._duration_from_payload(
                payload,
                DEFAULT_TASK_DURATION_S["draw_lab"],
            ),
            payload=payload,
            emsim_action=None,
            created_at_s=self.now_s,
        )

    def _schedule_lab_result(self, task: NurseTask) -> PendingEvent:
        test_name = self._lab_name(task.payload)
        result_payload = deepcopy(
            task.payload.get("result_payload", task.payload.get("result"))
        )
        event_payload = {"test_name": test_name}
        if result_payload is not None:
            event_payload["result"] = result_payload

        due_at_s = self.now_s + self._lab_turnaround_s(task.payload)
        return self.event_queue.schedule_event(
            event_type="lab_result_ready",
            due_at_s=due_at_s,
            payload=event_payload,
            source_order_id=task.source_order_id,
            created_at_s=self.now_s,
        )

    def _emsim_action(
        self,
        action_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        action_override = payload.get("emsim_action")
        if action_override is not None:
            return deepcopy(action_override)
        action = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in _WORKFLOW_PAYLOAD_KEYS
        }
        action["type"] = action_type
        return action

    def _duration_from_payload(
        self,
        payload: dict[str, Any],
        default_duration_s: float,
    ) -> float:
        return float(
            payload.get(
                "duration_s",
                payload.get("task_duration_s", default_duration_s),
            )
        )

    def _lab_turnaround_s(self, payload: dict[str, Any]) -> float:
        if "turnaround_s" in payload:
            return float(payload["turnaround_s"])
        return self.lab_turnaround_s.get(self._lab_name(payload), 600.0)

    @staticmethod
    def _lab_name(payload: dict[str, Any]) -> str:
        return str(payload.get("test_name", payload.get("name", "")))
