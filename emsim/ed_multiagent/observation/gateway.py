"""Engine-agnostic role observation gateway.

The gateway consumes only public multi-agent interfaces and returns filtered
views for each role. It must not depend on EMSim's rule engine, hidden state,
or debug state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from ..actions import ResponseOpportunityQueue
from ..memory import (
    ConversationMemory,
    DiscoveredClinicalMemory,
    GroundTruthMemory,
    PatientPrivateMemory,
    RelativePrivateMemory,
)
from ..world import WorkflowEngine
from .role_views import (
    ClinicianObservation,
    NurseObservation,
    PatientObservation,
    RelativeObservation,
)
from .subjective import SubjectiveState, subjective_from_observation


FORBIDDEN_OBSERVATION_KEYS = {
    "hidden_state",
    "mechanism",
    "pathology",
    "active_drug_effects",
    "_hidden_state",
    "CO_index",
    "SVR_index",
    "preload_index",
    "chronotropic_drive",
    "PaO2_effective",
    "shunt_fraction",
}

_INTERNAL_WORKFLOW_KEYS = {"emsim_action"}
_NUMERIC_VITAL_KEYS = {"HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"}
_TERMINAL_ORDER_STATUSES = {"completed", "failed", "cancelled"}


class ObservationGateway:
    """Build role-specific observations from public simulation surfaces."""

    def __init__(
        self,
        *,
        engine_session: Any,
        workflow_engine: WorkflowEngine | None = None,
        ground_truth_memory: GroundTruthMemory | None = None,
        discovered_memory: DiscoveredClinicalMemory | None = None,
        conversation_memory: ConversationMemory | None = None,
        patient_private_memory: PatientPrivateMemory | None = None,
        relative_private_memory: RelativePrivateMemory | None = None,
        subjective_state_provider: Any | None = None,
        subjective_state: SubjectiveState | Mapping[str, Any] | None = None,
        response_opportunities: ResponseOpportunityQueue | None = None,
    ):
        self.engine_session = engine_session
        self.workflow_engine = workflow_engine
        # Accepted for construction symmetry with scenario state, but never
        # used as an observation source.
        self.ground_truth_memory = ground_truth_memory
        self.discovered_memory = discovered_memory or DiscoveredClinicalMemory()
        self.conversation_memory = conversation_memory or ConversationMemory()
        self.patient_private_memory = patient_private_memory or PatientPrivateMemory()
        self.relative_private_memory = relative_private_memory or RelativePrivateMemory()
        self.subjective_state_provider = subjective_state_provider
        self.subjective_state = _coerce_subjective_state(subjective_state)
        self.response_opportunities = response_opportunities

    def for_clinician(self) -> ClinicianObservation:
        """Return the clinician view without hidden truth leakage."""

        discovered = self.discovered_memory
        return ClinicianObservation(
            time_s=self._current_time_s(),
            allergies_known=bool(discovered.allergies_known),
            allergies=_sanitize_dynamic(discovered.allergies)
            if discovered.allergies_known
            else [],
            medications_known=bool(discovered.medications_known),
            medications=_sanitize_dynamic(discovered.medications)
            if discovered.medications_known
            else [],
            pmh_known=bool(discovered.pmh_known),
            past_medical_history=_sanitize_dynamic(discovered.past_medical_history)
            if discovered.pmh_known
            else [],
            symptom_history=_sanitize_dynamic(discovered.symptom_history),
            discovered_vitals_history=_sanitize_dynamic(discovered.vitals_history),
            exam_findings=_sanitize_dynamic(discovered.exam_findings),
            available_test_results=_sanitize_dynamic(discovered.test_results),
            recent_dialogue=self._recent_dialogue(),
            response_opportunities=self._response_opportunities_for("clinician"),
        )

    def for_nurse(self) -> NurseObservation:
        """Return the nurse view over observable vitals and workflow state."""

        current_vitals = _sanitize_dynamic(self.engine_session.observe_vitals())
        return NurseObservation(
            time_s=self._current_time_s(),
            current_vitals=current_vitals,
            pending_orders=self._pending_orders(),
            active_task=self._active_task(),
            task_queue_summary=self._task_queue_summary(),
            queued_tasks=self._queued_tasks(),
            completed_tasks=self._completed_tasks(),
            pending_events=self._pending_events(),
            recent_dialogue=self._recent_dialogue(),
            response_opportunities=self._response_opportunities_for("nurse"),
        )

    def for_patient(self) -> PatientObservation:
        """Return the patient view without numeric vital signs by default."""

        return PatientObservation(
            time_s=self._current_time_s(),
            subjective_state=self._current_subjective_state(),
            patient_private_memory=_sanitize_patient_or_relative_dynamic(
                asdict(self.patient_private_memory)
            ),
            recent_dialogue=self._recent_dialogue(
                strip_structured_vitals=True,
            ),
            response_opportunities=self._response_opportunities_for(
                "patient",
                strip_structured_vitals=True,
            ),
            environment_context={},
        )

    def for_relative(self) -> RelativeObservation:
        """Return the relative view from visible state and relative memory."""

        subjective = self._current_subjective_state()
        return RelativeObservation(
            time_s=self._current_time_s(),
            visible_patient_state=_visible_patient_state(subjective),
            relative_private_memory=_sanitize_patient_or_relative_dynamic(
                asdict(self.relative_private_memory)
            ),
            emotional_or_social_context=(
                _sanitize_patient_or_relative_dynamic(
                    self.conversation_memory.emotional_or_social_summary
                )
            ),
            recent_dialogue=self._recent_dialogue(
                strip_structured_vitals=True,
            ),
            response_opportunities=self._response_opportunities_for(
                "relative",
                strip_structured_vitals=True,
            ),
        )

    def for_role(
        self,
        role: str,
    ) -> (
        ClinicianObservation
        | NurseObservation
        | PatientObservation
        | RelativeObservation
    ):
        """Return the observation for a supported role name."""

        normalized = role.strip().lower()
        if normalized == "clinician":
            return self.for_clinician()
        if normalized == "nurse":
            return self.for_nurse()
        if normalized == "patient":
            return self.for_patient()
        if normalized == "relative":
            return self.for_relative()
        raise ValueError(f"unsupported observation role: {role!r}")

    def _current_time_s(self) -> float | None:
        observation = self.engine_session.observe()
        if isinstance(observation, Mapping) and "time_s" in observation:
            return float(observation["time_s"])
        now_s = getattr(self.engine_session, "now_s", None)
        if now_s is None:
            return None
        return float(now_s)

    def _current_subjective_state(self) -> SubjectiveState | None:
        if self.subjective_state is not None:
            return deepcopy(self.subjective_state)

        provider = self.subjective_state_provider
        if provider is not None:
            provided = _call_subjective_provider(provider)
            return _coerce_subjective_state(provided)

        vitals = self.engine_session.observe_vitals()
        return subjective_from_observation(vitals)

    def _recent_dialogue(
        self,
        *,
        strip_structured_vitals: bool = False,
    ) -> list[dict[str, Any]]:
        if strip_structured_vitals:
            return _sanitize_patient_or_relative_dynamic(
                self.conversation_memory.recent_window
            )
        return _sanitize_dynamic(self.conversation_memory.recent_window)

    def _response_opportunities_for(
        self,
        role: str,
        *,
        strip_structured_vitals: bool = False,
    ) -> list[dict[str, Any]]:
        if self.response_opportunities is None:
            return []
        opportunities = self.response_opportunities.list_open(role)
        if strip_structured_vitals:
            return _sanitize_patient_or_relative_dynamic(opportunities)
        return _sanitize_dynamic(opportunities)

    def _pending_orders(self) -> list[dict[str, Any]]:
        if self.workflow_engine is None:
            return []
        return [
            _order_view(order)
            for order in self.workflow_engine.order_manager.list_orders()
            if order.status not in _TERMINAL_ORDER_STATUSES
        ]

    def _active_task(self) -> dict[str, Any] | None:
        if self.workflow_engine is None:
            return None
        active = self.workflow_engine.nurse_task_queue.active_tasks()
        if not active:
            return None
        return _task_view(active[0])

    def _queued_tasks(self) -> list[dict[str, Any]]:
        if self.workflow_engine is None:
            return []
        return [
            _task_view(task)
            for task in self.workflow_engine.nurse_task_queue.queued_tasks()
        ]

    def _completed_tasks(self) -> list[dict[str, Any]]:
        if self.workflow_engine is None:
            return []
        return [
            _task_view(task)
            for task in self.workflow_engine.nurse_task_queue.list_tasks("done")
        ]

    def _pending_events(self) -> list[dict[str, Any]]:
        if self.workflow_engine is None:
            return []
        return _sanitize_dynamic(self.workflow_engine.pending_events)

    def _task_queue_summary(self) -> dict[str, Any]:
        if self.workflow_engine is None:
            return {
                "active_count": 0,
                "queued_count": 0,
                "completed_count": 0,
                "failed_count": 0,
            }
        queue = self.workflow_engine.nurse_task_queue
        return {
            "active_count": len(queue.active_tasks()),
            "queued_count": len(queue.queued_tasks()),
            "completed_count": len(queue.list_tasks("done")),
            "failed_count": len(queue.list_tasks("failed")),
        }


def _order_view(order: Any) -> dict[str, Any]:
    return _sanitize_dynamic(
        {
            "order_id": order.order_id,
            "ordered_by": order.ordered_by,
            "order_type": order.order_type,
            "payload": order.payload,
            "priority": order.priority,
            "status": order.status,
            "ordered_at_s": order.ordered_at_s,
            "started_at_s": order.started_at_s,
            "completed_at_s": order.completed_at_s,
        }
    )


def _task_view(task: Any) -> dict[str, Any]:
    return _sanitize_dynamic(
        {
            "task_id": task.task_id,
            "source_order_id": task.source_order_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "duration_s": task.duration_s,
            "remaining_s": task.remaining_s,
            "status": task.status,
            "payload": task.payload,
            "created_at_s": task.created_at_s,
            "started_at_s": task.started_at_s,
            "completed_at_s": task.completed_at_s,
        }
    )


def _visible_patient_state(subjective: SubjectiveState | None) -> dict[str, Any]:
    if subjective is None:
        return {}
    return {
        "visible_distress": subjective.visible_distress,
        "speech_capacity": str(subjective.speech_capacity),
        "appears_short_of_breath": subjective.dyspnea_severity >= 0.35,
        "appears_confused": subjective.confusion_level >= 0.35,
        "appears_dizzy": subjective.dizziness >= 0.35,
        "pain_distress": subjective.pain_distress,
    }


def _call_subjective_provider(
    provider: Callable[[], SubjectiveState | Mapping[str, Any]] | Any,
) -> SubjectiveState | Mapping[str, Any] | None:
    if callable(provider):
        return provider()
    for method_name in (
        "get_subjective_state",
        "current_subjective_state",
        "subjective_state",
    ):
        method = getattr(provider, method_name, None)
        if callable(method):
            return method()
    if isinstance(provider, (SubjectiveState, Mapping)):
        return provider
    raise TypeError("subjective_state_provider must be callable or expose a state method")


def _coerce_subjective_state(
    value: SubjectiveState | Mapping[str, Any] | None,
) -> SubjectiveState | None:
    if value is None:
        return None
    if isinstance(value, SubjectiveState):
        return deepcopy(value)
    if isinstance(value, Mapping):
        return SubjectiveState(**_sanitize_dynamic(value))
    raise TypeError("subjective_state must be a SubjectiveState or mapping")


def _sanitize_dynamic(value: Any) -> Any:
    return _sanitize_with_forbidden_keys(value, extra_forbidden_keys=set())


def _sanitize_patient_or_relative_dynamic(value: Any) -> Any:
    return _sanitize_with_forbidden_keys(
        value,
        extra_forbidden_keys=_NUMERIC_VITAL_KEYS,
    )


def _sanitize_with_forbidden_keys(
    value: Any,
    *,
    extra_forbidden_keys: set[str],
) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_with_forbidden_keys(
            asdict(value),
            extra_forbidden_keys=extra_forbidden_keys,
        )
    if isinstance(value, Mapping):
        sanitized: dict[Any, Any] = {}
        for key, nested in value.items():
            if isinstance(key, str) and (
                key in FORBIDDEN_OBSERVATION_KEYS
                or key in _INTERNAL_WORKFLOW_KEYS
                or key in extra_forbidden_keys
            ):
                continue
            sanitized[deepcopy(key)] = _sanitize_with_forbidden_keys(
                nested,
                extra_forbidden_keys=extra_forbidden_keys,
            )
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_with_forbidden_keys(
                item,
                extra_forbidden_keys=extra_forbidden_keys,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_with_forbidden_keys(
                item,
                extra_forbidden_keys=extra_forbidden_keys,
            )
            for item in value
        )
    if isinstance(value, set):
        return {
            _sanitize_with_forbidden_keys(
                item,
                extra_forbidden_keys=extra_forbidden_keys,
            )
            for item in value
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _sanitize_with_forbidden_keys(
                item,
                extra_forbidden_keys=extra_forbidden_keys,
            )
            for item in value
        ]
    return deepcopy(value)


__all__ = [
    "FORBIDDEN_OBSERVATION_KEYS",
    "ObservationGateway",
]
