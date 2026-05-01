"""Role-specific observation view models.

These dataclasses are deliberately plain containers. The gateway owns the
filtering rules that decide what can enter each view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .subjective import SubjectiveState


@dataclass
class ClinicianObservation:
    """Clinician-facing state assembled from discovered clinical facts only."""

    role: str = "clinician"
    time_s: float | None = None
    allergies_known: bool = False
    allergies: list[str] = field(default_factory=list)
    medications_known: bool = False
    medications: list[str] = field(default_factory=list)
    pmh_known: bool = False
    past_medical_history: list[str] = field(default_factory=list)
    symptom_history: dict[str, Any] = field(default_factory=dict)
    discovered_vitals_history: list[dict[str, Any]] = field(default_factory=list)
    exam_findings: dict[str, Any] = field(default_factory=dict)
    available_test_results: dict[str, Any] = field(default_factory=dict)
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    response_opportunities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NurseObservation:
    """Nurse-facing state for observable vitals and workflow execution."""

    role: str = "nurse"
    time_s: float | None = None
    current_vitals: dict[str, Any] = field(default_factory=dict)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)
    active_task: dict[str, Any] | None = None
    task_queue_summary: dict[str, Any] = field(default_factory=dict)
    queued_tasks: list[dict[str, Any]] = field(default_factory=list)
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    response_opportunities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PatientObservation:
    """Patient-facing state for subjective experience and private memory."""

    role: str = "patient"
    time_s: float | None = None
    subjective_state: SubjectiveState | None = None
    patient_private_memory: dict[str, Any] = field(default_factory=dict)
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    response_opportunities: list[dict[str, Any]] = field(default_factory=list)
    environment_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelativeObservation:
    """Relative-facing state for visible patient status and family context."""

    role: str = "relative"
    time_s: float | None = None
    visible_patient_state: dict[str, Any] = field(default_factory=dict)
    relative_private_memory: dict[str, Any] = field(default_factory=dict)
    emotional_or_social_context: str | None = None
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    response_opportunities: list[dict[str, Any]] = field(default_factory=list)


__all__ = [
    "ClinicianObservation",
    "NurseObservation",
    "PatientObservation",
    "RelativeObservation",
]
