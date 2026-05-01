"""Pending workflow events such as delayed lab results."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingEvent:
    """Workflow event scheduled for a future simulated timestamp."""

    event_id: str = ""
    event_type: str = "workflow_event"
    due_at_s: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    source_order_id: str | None = None
    status: str = "pending"
    created_at_s: float = 0.0
    delivered_at_s: float | None = None

    def __post_init__(self) -> None:
        self.due_at_s = float(self.due_at_s)
        self.created_at_s = float(self.created_at_s)
        if self.delivered_at_s is not None:
            self.delivered_at_s = float(self.delivered_at_s)
        self.payload = deepcopy(self.payload)

    def mark_delivered(self, now_s: float) -> None:
        self.status = "delivered"
        self.delivered_at_s = float(now_s)


class PendingEventQueue:
    """In-memory queue for delayed deterministic workflow events.

    Delivered events are retained with status="delivered" for audit/replay.
    """

    def __init__(
        self,
        events: list[PendingEvent] | None = None,
        *,
        now_s: float = 0.0,
    ):
        self.events: dict[str, PendingEvent] = {}
        self.now_s = float(now_s)
        self._next_event_number = 1
        for event in events or []:
            self.add_event(event)

    def schedule_event(
        self,
        *,
        event_type: str,
        due_at_s: float | None = None,
        delay_s: float | None = None,
        now_s: float | None = None,
        payload: dict[str, Any] | None = None,
        source_order_id: str | None = None,
        created_at_s: float = 0.0,
        event_id: str | None = None,
    ) -> PendingEvent:
        if due_at_s is None:
            if delay_s is None:
                raise ValueError("due_at_s or delay_s is required")
            base_time_s = self.now_s if now_s is None else float(now_s)
            due_at_s = base_time_s + float(delay_s)
        event = PendingEvent(
            event_id=event_id or self._make_event_id(),
            event_type=event_type,
            due_at_s=due_at_s,
            payload=payload or {},
            source_order_id=source_order_id,
            created_at_s=created_at_s,
        )
        return self.add_event(event)

    def add_event(self, event: PendingEvent | dict[str, Any]) -> PendingEvent:
        if isinstance(event, dict):
            return self.schedule_event(**event)
        if not event.event_id:
            event.event_id = self._make_event_id()
        if event.event_id in self.events:
            raise ValueError(f"duplicate event_id {event.event_id!r}")
        self.events[event.event_id] = event
        return event

    def get_event(self, event_id: str) -> PendingEvent:
        return self.events[event_id]

    def pending_events(self) -> list[PendingEvent]:
        return [event for event in self.events.values() if event.status == "pending"]

    def pop_due(self, now_s: float) -> list[PendingEvent]:
        due = [
            event
            for event in self.pending_events()
            if event.due_at_s <= float(now_s)
        ]
        due.sort(key=lambda event: (event.due_at_s, event.event_id))
        for event in due:
            event.mark_delivered(now_s)
        return due

    def advance(self, dt_s: float) -> list[PendingEvent]:
        dt_s = float(dt_s)
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        self.now_s += dt_s
        return self.pop_due(self.now_s)

    def _make_event_id(self) -> str:
        event_id = f"event-{self._next_event_number}"
        self._next_event_number += 1
        return event_id
