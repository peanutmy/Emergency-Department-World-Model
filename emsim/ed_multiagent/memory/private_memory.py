"""Role-private memory for patient and relative personas."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatientPrivateMemory:
    """Facts and persona data available to the patient role."""

    personality: str | None = None
    cooperation_baseline: float = 1.0
    health_literacy: str | None = None
    symptom_knowledge: dict[str, Any] = field(default_factory=dict)
    sensitive_topics: list[str] = field(default_factory=list)
    symptom_timeline: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.cooperation_baseline = float(self.cooperation_baseline)
        self.symptom_knowledge = deepcopy(self.symptom_knowledge)
        self.sensitive_topics = deepcopy(self.sensitive_topics)
        self.symptom_timeline = deepcopy(self.symptom_timeline)


@dataclass
class RelativePrivateMemory:
    """Facts and persona data available to the relative role."""

    relationship: str | None = None
    knows_allergies: bool = False
    knows_medications: bool = False
    knows_recent_events: bool = False
    personality: str | None = None
    private_facts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.private_facts = deepcopy(self.private_facts)
