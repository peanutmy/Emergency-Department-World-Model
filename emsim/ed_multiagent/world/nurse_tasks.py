"""Nurse task queue for delayed workflow execution."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


TaskType = Literal[
    "administer_drug",
    "apply_intervention",
    "draw_lab",
    "measure_vitals",
    "assist_procedure",
    "report_result",
]
TaskStatus = Literal["queued", "active", "done", "failed"]

MAX_ACTIVE_TASKS_PER_NURSE = 1

DEFAULT_TASK_DURATION_S: dict[str, float] = {
    "administer_drug": 30.0,
    "apply_intervention": 30.0,
    "draw_lab": 90.0,
    "measure_vitals": 30.0,
    "assist_procedure": 120.0,
    "report_result": 15.0,
}

INTERVENTION_TASK_DURATION_S: dict[str, float] = {
    "intubate": 180.0,
    "apply_NRB": 30.0,
    "apply_nasal": 30.0,
    "apply_BVM": 30.0,
    "apply_BiPAP": 30.0,
    "defibrillate": 30.0,
    "start_CPR": 30.0,
    "give_fluids": 120.0,
}


@dataclass
class NurseTask:
    """
    duration_s and remaining_s are measured in simulated seconds,
    not number of simulation rounds.

    Every round must decrement remaining_s by the actual dt_s used
    for that round.

    This keeps task duration stable across SimMode changes.
    """

    task_id: str = ""
    source_order_id: str | None = None
    task_type: str = "administer_drug"
    priority: str = "stat"
    duration_s: float = 30.0
    remaining_s: float | None = None
    status: str = "queued"
    payload: dict[str, Any] = field(default_factory=dict)
    emsim_action: dict[str, Any] | None = None
    created_at_s: float = 0.0
    started_at_s: float | None = None
    completed_at_s: float | None = None

    def __post_init__(self) -> None:
        self.duration_s = float(self.duration_s)
        if self.remaining_s is None:
            self.remaining_s = self.duration_s
        else:
            self.remaining_s = float(self.remaining_s)
        self.payload = deepcopy(self.payload)
        self.emsim_action = deepcopy(self.emsim_action)
        self.created_at_s = float(self.created_at_s)
        if self.started_at_s is not None:
            self.started_at_s = float(self.started_at_s)
        if self.completed_at_s is not None:
            self.completed_at_s = float(self.completed_at_s)

    def start(self, now_s: float | None = None) -> None:
        if self.status != "queued":
            return
        self.status = "active"
        if now_s is not None:
            self.started_at_s = float(now_s)

    def advance(self, dt_s: float, now_s: float | None = None) -> bool:
        """Advance an active task by simulated seconds and return completion."""

        dt_s = float(dt_s)
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        if self.status != "active":
            return False

        self.remaining_s = max(0.0, float(self.remaining_s or 0.0) - dt_s)
        if self.remaining_s == 0.0:
            self.status = "done"
            if now_s is not None:
                self.completed_at_s = float(now_s)
            return True
        return False

    def complete(self, now_s: float | None = None) -> dict[str, Any] | None:
        self.remaining_s = 0.0
        self.status = "done"
        if now_s is not None:
            self.completed_at_s = float(now_s)
        return deepcopy(self.emsim_action)


class NurseTaskQueue:
    """Single-nurse FIFO queue with simulated-second task timing."""

    def __init__(
        self,
        tasks: list[NurseTask] | None = None,
        *,
        max_active_tasks: int = MAX_ACTIVE_TASKS_PER_NURSE,
    ):
        if max_active_tasks < 1:
            raise ValueError("max_active_tasks must be at least 1")
        self.tasks: dict[str, NurseTask] = {}
        self.max_active_tasks = max_active_tasks
        self._next_task_number = 1
        for task in tasks or []:
            self.add_task(task)

    def create_task(
        self,
        *,
        task_type: str,
        source_order_id: str | None = None,
        priority: str = "stat",
        duration_s: float | None = None,
        payload: dict[str, Any] | None = None,
        emsim_action: dict[str, Any] | None = None,
        created_at_s: float = 0.0,
        task_id: str | None = None,
    ) -> NurseTask:
        task = NurseTask(
            task_id=task_id or self._make_task_id(),
            source_order_id=source_order_id,
            task_type=task_type,
            priority=priority,
            duration_s=duration_s
            if duration_s is not None
            else DEFAULT_TASK_DURATION_S.get(task_type, 30.0),
            payload=payload or {},
            emsim_action=emsim_action,
            created_at_s=created_at_s,
        )
        return self.add_task(task)

    def add_task(self, task: NurseTask | dict[str, Any]) -> NurseTask:
        if isinstance(task, dict):
            return self.create_task(**task)
        if not task.task_id:
            task.task_id = self._make_task_id()
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task_id {task.task_id!r}")
        self.tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> NurseTask:
        return self.tasks[task_id]

    def list_tasks(self, status: str | None = None) -> list[NurseTask]:
        if status is None:
            return list(self.tasks.values())
        return [task for task in self.tasks.values() if task.status == status]

    def active_tasks(self) -> list[NurseTask]:
        return self.list_tasks("active")

    def queued_tasks(self) -> list[NurseTask]:
        return self.list_tasks("queued")

    def activate_queued(self, now_s: float | None = None) -> list[NurseTask]:
        """Activate queued work in M1b FIFO order.

        M1b activates queued tasks in FIFO insertion order; priority is stored
        but not consulted until a future prioritization milestone.
        """

        activated: list[NurseTask] = []
        open_slots = self.max_active_tasks - len(self.active_tasks())
        if open_slots <= 0:
            return activated

        for task in self.queued_tasks()[:open_slots]:
            task.start(now_s=now_s)
            activated.append(task)
        return activated

    def advance_active(self, dt_s: float, now_s: float | None = None) -> list[NurseTask]:
        completed: list[NurseTask] = []
        for task in list(self.active_tasks()):
            if task.advance(dt_s, now_s=now_s):
                completed.append(task)
        return completed

    def advance(self, dt_s: float, now_s: float | None = None) -> list[NurseTask]:
        """Activate queued work up to capacity, then decrement by ``dt_s``."""

        self.activate_queued(now_s=now_s)
        return self.advance_active(dt_s, now_s=now_s)

    def _make_task_id(self) -> str:
        task_id = f"task-{self._next_task_number}"
        self._next_task_number += 1
        return task_id
