"""Clinician-facing structured clinical memory."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredClinicalMemory:
    """Clinical facts currently known to the care team.

    Ground-truth facts are copied here only when explicitly discovered through
    a structured update path.
    """

    allergies_known: bool = False
    allergies: list[str] = field(default_factory=list)
    medications_known: bool = False
    medications: list[str] = field(default_factory=list)
    pmh_known: bool = False
    past_medical_history: list[str] = field(default_factory=list)
    symptom_history: dict[str, Any] = field(default_factory=dict)
    vitals_history: list[dict[str, Any]] = field(default_factory=list)
    exam_findings: dict[str, Any] = field(default_factory=dict)
    test_results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allergies = deepcopy(self.allergies)
        self.medications = deepcopy(self.medications)
        self.past_medical_history = deepcopy(self.past_medical_history)
        self.symptom_history = deepcopy(self.symptom_history)
        self.vitals_history = deepcopy(self.vitals_history)
        self.exam_findings = deepcopy(self.exam_findings)
        self.test_results = deepcopy(self.test_results)

    def discover_allergies(self, allergies: list[str]) -> None:
        self.allergies_known = True
        self.allergies = deepcopy(allergies)

    def discover_medications(self, medications: list[str]) -> None:
        self.medications_known = True
        self.medications = deepcopy(medications)

    def discover_past_medical_history(self, items: list[str]) -> None:
        self.pmh_known = True
        self.past_medical_history = deepcopy(items)

    def update_symptom_history(self, slots: dict[str, Any]) -> None:
        """M1c uses shallow slot updates; nested merge is intentionally deferred."""

        self.symptom_history.update(deepcopy(slots))

    def add_vitals(self, snapshot: dict[str, Any]) -> None:
        self.vitals_history.append(deepcopy(snapshot))

    def update_exam_findings(self, findings: dict[str, Any]) -> None:
        self.exam_findings.update(deepcopy(findings))

    def add_test_result(self, test_name: str, result: dict[str, Any]) -> None:
        self.test_results[str(test_name)] = deepcopy(result)
