"""Agent activation policy for dynamic simulation modes.

These helpers decide whether to call optional cognitive/verbal agents. They do
not affect deterministic physiology or workflow execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .sim_mode import SimMode, _coerce_mode, _event_kind


@dataclass(frozen=True)
class AgentActivationPolicy:
    code_clinician_cadence_s: float = 120.0
    urgent_clinician_cadence_s: float = 30.0
    code_relative_throttle_s: float = 30.0

    def should_call_clinician_agent(
        self,
        mode: SimMode | str,
        now_s: float,
        last_clinician_act_s: float | None,
        pending_decision_event: bool,
        rhythm_check_due: bool,
        nurse_report_pending: bool,
        medication_window_due: bool = False,
    ) -> bool:
        if pending_decision_event:
            return True
        if rhythm_check_due:
            return True
        if nurse_report_pending:
            return True
        if medication_window_due:
            return True
        if last_clinician_act_s is None:
            return True

        mode = _coerce_mode(mode)
        elapsed_s = float(now_s) - float(last_clinician_act_s)
        if mode == SimMode.CODE:
            return elapsed_s >= self.code_clinician_cadence_s
        if mode == SimMode.URGENT:
            return elapsed_s >= self.urgent_clinician_cadence_s
        return True

    def should_call_nurse_agent(
        self,
        mode: SimMode | str,
        nurse_state: Any | None = None,
        event: Any | None = None,
        *,
        nurse_report_pending: bool = False,
        requires_nurse_report: bool = False,
        needs_clarification: bool = False,
        warning_pending: bool = False,
        major_clinical_change: bool = False,
    ) -> bool:
        # mode accepted for symmetry with other activation policies; nurse verbalization is event-driven.
        _ = mode
        explicit_need = any(
            (
                nurse_report_pending,
                requires_nurse_report,
                needs_clarification,
                warning_pending,
                major_clinical_change,
            )
        )
        if explicit_need:
            return True

        if _any_truthy_field(
            nurse_state,
            (
                "nurse_report_pending",
                "requires_nurse_report",
                "needs_clarification",
                "warning_pending",
            ),
        ):
            return True

        if _any_truthy_field(
            event,
            (
                "nurse_report_pending",
                "requires_nurse_report",
                "major_clinical_change",
                "warning_pending",
            ),
        ):
            return True

        return _event_kind(event) in {
            "warning",
            "nurse_warning",
            "major_clinical_change",
            "clinical_deterioration",
            "lab_result_ready",
        }

    def should_call_patient_agent(
        self,
        mode: SimMode | str,
        speech_capacity: Any | None = None,
        response_opportunity: Any | None = None,
        *,
        has_response_opportunity: bool = False,
    ) -> bool:
        if _has_response_opportunity(response_opportunity, has_response_opportunity):
            return True

        mode = _coerce_mode(mode)
        capacity = _speech_capacity_value(speech_capacity)
        if mode == SimMode.CODE and capacity == "unable":
            return False
        return True

    def should_call_relative_agent(
        self,
        mode: SimMode | str,
        now_s: float,
        last_relative_act_s: float | None,
        major_event: bool = False,
    ) -> bool:
        if major_event:
            return True
        if last_relative_act_s is None:
            return True

        mode = _coerce_mode(mode)
        if mode == SimMode.CODE:
            elapsed_s = float(now_s) - float(last_relative_act_s)
            return elapsed_s >= self.code_relative_throttle_s
        return True


_DEFAULT_POLICY = AgentActivationPolicy()


def should_call_clinician_agent(
    mode: SimMode | str,
    now_s: float,
    last_clinician_act_s: float | None,
    pending_decision_event: bool,
    rhythm_check_due: bool,
    nurse_report_pending: bool,
    medication_window_due: bool = False,
) -> bool:
    return _DEFAULT_POLICY.should_call_clinician_agent(
        mode,
        now_s,
        last_clinician_act_s,
        pending_decision_event,
        rhythm_check_due,
        nurse_report_pending,
        medication_window_due,
    )


def should_call_nurse_agent(
    mode: SimMode | str,
    nurse_state: Any | None = None,
    event: Any | None = None,
    *,
    nurse_report_pending: bool = False,
    requires_nurse_report: bool = False,
    needs_clarification: bool = False,
    warning_pending: bool = False,
    major_clinical_change: bool = False,
) -> bool:
    return _DEFAULT_POLICY.should_call_nurse_agent(
        mode,
        nurse_state,
        event,
        nurse_report_pending=nurse_report_pending,
        requires_nurse_report=requires_nurse_report,
        needs_clarification=needs_clarification,
        warning_pending=warning_pending,
        major_clinical_change=major_clinical_change,
    )


def should_call_patient_agent(
    mode: SimMode | str,
    speech_capacity: Any | None = None,
    response_opportunity: Any | None = None,
    *,
    has_response_opportunity: bool = False,
) -> bool:
    return _DEFAULT_POLICY.should_call_patient_agent(
        mode,
        speech_capacity,
        response_opportunity,
        has_response_opportunity=has_response_opportunity,
    )


def should_call_relative_agent(
    mode: SimMode | str,
    now_s: float,
    last_relative_act_s: float | None,
    major_event: bool = False,
) -> bool:
    return _DEFAULT_POLICY.should_call_relative_agent(
        mode,
        now_s,
        last_relative_act_s,
        major_event,
    )


def _has_response_opportunity(
    response_opportunity: Any | None,
    has_response_opportunity: bool,
) -> bool:
    if has_response_opportunity:
        return True
    if isinstance(response_opportunity, bool):
        return response_opportunity
    return response_opportunity is not None


def _speech_capacity_value(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Mapping):
        capacity = value.get("speech_capacity")
        return None if capacity is None else str(capacity).strip().lower()
    if hasattr(value, "speech_capacity"):
        capacity = getattr(value, "speech_capacity")
        return None if capacity is None else str(capacity).strip().lower()
    return str(value).strip().lower()


def _any_truthy_field(value: Any | None, names: tuple[str, ...]) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(bool(value.get(name)) for name in names)
    return any(bool(getattr(value, name, False)) for name in names)


__all__ = [
    "AgentActivationPolicy",
    "should_call_clinician_agent",
    "should_call_nurse_agent",
    "should_call_patient_agent",
    "should_call_relative_agent",
]
