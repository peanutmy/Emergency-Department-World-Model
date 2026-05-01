"""Minimal deterministic round logging for no-LLM orchestration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


@dataclass
class RoundLogEntry:
    """Replay-oriented summary of one deterministic orchestrator round."""

    round_index: int
    now_s: float
    sim_mode: str
    dt_s: float
    clinician_turn: dict[str, Any]
    submitted_orders: list[dict[str, Any]] = field(default_factory=list)
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    applied_emsim_actions: list[dict[str, Any]] = field(default_factory=list)
    workflow_events: list[dict[str, Any]] = field(default_factory=list)
    physiology_events: list[dict[str, Any]] = field(default_factory=list)
    memory_deltas: dict[str, Any] = field(default_factory=dict)
    response_opportunities_created: list[dict[str, Any]] = field(default_factory=list)
    response_opportunities_resolved: list[dict[str, Any]] = field(default_factory=list)


class RoundLogger:
    """In-memory logger used by deterministic tests and future replay tools."""

    def __init__(self) -> None:
        self.entries: list[RoundLogEntry] = []

    def log_round(self, round_result: Any) -> RoundLogEntry:
        entry = RoundLogEntry(
            round_index=len(self.entries),
            now_s=float(getattr(round_result, "now_s")),
            sim_mode=_enum_or_value(getattr(round_result, "sim_mode")),
            dt_s=float(getattr(round_result, "dt_s")),
            clinician_turn=_to_plain(getattr(round_result, "clinician_turn")),
            submitted_orders=_to_plain(getattr(round_result, "submitted_orders", [])),
            completed_tasks=_to_plain(
                getattr(getattr(round_result, "workflow_result", None), "completed_tasks", [])
            ),
            applied_emsim_actions=_to_plain(
                getattr(round_result, "applied_emsim_actions", [])
            ),
            workflow_events=_to_plain(getattr(round_result, "fired_events", [])),
            physiology_events=_to_plain(
                getattr(getattr(round_result, "engine_result", None), "events", [])
            ),
            memory_deltas=_to_plain(getattr(round_result, "memory_deltas", {})),
            response_opportunities_created=_to_plain(
                getattr(round_result, "response_opportunities_created", [])
            ),
            response_opportunities_resolved=_to_plain(
                getattr(round_result, "response_opportunities_resolved", [])
            ),
        )
        self.entries.append(entry)
        return entry


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            deepcopy(key): _to_plain(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    if isinstance(value, set):
        return {_to_plain(item) for item in value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain(item) for item in value]
    return deepcopy(value)


def _enum_or_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


__all__ = ["RoundLogEntry", "RoundLogger"]
