"""Scenario truth memory for the EMSim multi-agent layer."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundTruthMemory:
    """True patient facts that are not automatically clinician-visible."""

    allergies: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    past_medical_history: list[str] = field(default_factory=list)
    symptom_truth: dict[str, Any] = field(default_factory=dict)
    family_known_info: dict[str, Any] = field(default_factory=dict)
    diagnosis: str | None = None
    pathology: str | None = None

    def __post_init__(self) -> None:
        self.allergies = deepcopy(self.allergies)
        self.medications = deepcopy(self.medications)
        self.past_medical_history = deepcopy(self.past_medical_history)
        self.symptom_truth = deepcopy(self.symptom_truth)
        self.family_known_info = deepcopy(self.family_known_info)
