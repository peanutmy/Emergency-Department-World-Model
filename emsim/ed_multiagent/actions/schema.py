"""Shared action schemas for multi-agent turn outputs."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTurnOutput:
    """Structured output envelope returned by an agent for one turn.

    M5 keeps this deliberately small. The fields are intents or reports for
    orchestration layers to consume later; they do not execute physiology.
    For M5, each turn emits at most one information_action and one
    physical_action; later orchestrator integration may widen this to lists.
    """

    verbal_action: Any | None = None
    information_action: dict[str, Any] | None = None
    physical_action: dict[str, Any] | None = None
    meta_action: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.verbal_action = deepcopy(self.verbal_action)
        self.information_action = _copy_optional_mapping(
            self.information_action,
            "information_action",
        )
        self.physical_action = _copy_optional_mapping(
            self.physical_action,
            "physical_action",
        )
        self.meta_action = deepcopy(self.meta_action)
        self.metadata = _copy_mapping(self.metadata, "metadata")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentTurnOutput":
        return cls(**deepcopy(dict(value)))

    @property
    def information_actions(self) -> list[dict[str, Any]]:
        if self.information_action is None:
            return []
        return [deepcopy(self.information_action)]

    @property
    def physical_actions(self) -> list[dict[str, Any]]:
        if self.physical_action is None:
            return []
        return [deepcopy(self.physical_action)]


def _copy_optional_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _copy_mapping(value, field_name)


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return deepcopy(dict(value))


__all__ = ["AgentTurnOutput"]
