"""Stateful simulation mode management for the multi-agent orchestrator.

This module deliberately depends only on plain vitals dictionaries, optional
rhythm strings, duck-typed physiology events, and clinician meta events.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any


class SimMode(str, Enum):
    STABLE = "stable"
    URGENT = "urgent"
    CODE = "code"


SIM_MODE_PRIORITY: dict[SimMode, int] = {
    SimMode.STABLE: 0,
    SimMode.URGENT: 1,
    SimMode.CODE: 2,
}

DECLARE_CODE = "declare_code"
EXIT_CODE = "exit_code"


class ClinicianMetaEvent(str, Enum):
    DECLARE_CODE = DECLARE_CODE
    EXIT_CODE = EXIT_CODE


_ARREST_RHYTHMS = {
    "vf",
    "vfib",
    "v_fib",
    "ventricular_fibrillation",
    "pea",
    "pulseless_electrical_activity",
    "asystole",
    "asystolic",
    "pulseless_vt",
    "pulseless_vtach",
    "pulseless_ventricular_tachycardia",
}

_CODE_PHYSIOLOGY_EVENTS = {
    "arrest",
    "cardiac_arrest",
    "respiratory_arrest",
    "code_required",
    "severe_hypoxia",
}

_ROSC_EVENTS = {
    "rosc",
    "rosc_detected",
    "return_of_spontaneous_circulation",
}

_URGENT_PHYSIOLOGY_EVENTS = {
    "clinical_deterioration",
    "deterioration",
    "hypotension",
    "severe_tachycardia",
    "severe_bradycardia",
}


def determine_target_mode(
    vitals: Mapping[str, Any] | None,
    rhythm: str | None = None,
    clinician_event: Any | None = None,
    *,
    physiology_events: Iterable[Any] | None = None,
    clinician_meta_events: Iterable[Any] | None = None,
    clinician_declared_code: bool = False,
) -> SimMode:
    """Return the immediate target mode from engine-agnostic observations."""

    events = list(physiology_events or [])
    meta_names = _collect_meta_event_names(clinician_event, clinician_meta_events)

    if clinician_declared_code or _normalize_token(DECLARE_CODE) in meta_names:
        return SimMode.CODE

    if _is_arrest_rhythm(rhythm):
        return SimMode.CODE

    if any(_event_suggests_code(event) for event in events):
        return SimMode.CODE

    o2_sat = _vital_number(
        vitals,
        "O2Sat",
        "O2_sat",
        "SpO2",
        "oxygen_saturation",
    )
    if o2_sat is not None and o2_sat < 85:
        return SimMode.CODE

    hr = _vital_number(vitals, "HR", "heart_rate")
    if hr is not None and (hr > 180 or hr < 40):
        return SimMode.URGENT

    bp_sys = _vital_number(vitals, "BP_sys", "SBP", "systolic_bp")
    if bp_sys is not None and bp_sys < 80:
        return SimMode.URGENT

    if any(_event_suggests_urgent(event) for event in events):
        return SimMode.URGENT

    return SimMode.STABLE


class SimModeManager:
    """Stateful mode manager with immediate upgrades and hysteretic downgrades."""

    def __init__(
        self,
        downgrade_required_rounds: int = 3,
        *,
        initial_mode: SimMode | str = SimMode.STABLE,
    ):
        if downgrade_required_rounds < 1:
            raise ValueError("downgrade_required_rounds must be at least 1")
        self.current_mode = _coerce_mode(initial_mode)
        self.downgrade_required_rounds = int(downgrade_required_rounds)
        self.downgrade_counter = 0

    def update(
        self,
        vitals: Mapping[str, Any] | None = None,
        rhythm: str | None = None,
        clinician_event: Any | None = None,
        *,
        physiology_events: Iterable[Any] | None = None,
        clinician_meta_events: Iterable[Any] | None = None,
        rosc_achieved: bool = False,
        clinician_declared_code: bool = False,
        exit_code_requested: bool = False,
        code_downgrade_allowed: bool = False,
        code_downgrade_target: SimMode | str | None = None,
    ) -> SimMode:
        """Update and return the persistent simulation mode.

        CODE mode is intentionally sticky. It exits only after ROSC plus an
        explicit exit-code request, or when a caller provides an explicit
        scenario-specific ``code_downgrade_allowed`` signal.
        """

        events = list(physiology_events or [])
        meta_names = _collect_meta_event_names(clinician_event, clinician_meta_events)
        exit_requested = (
            exit_code_requested or _normalize_token(EXIT_CODE) in meta_names
        )
        rosc_now = rosc_achieved or any(_event_suggests_rosc(event) for event in events)

        target_mode = determine_target_mode(
            vitals,
            rhythm,
            clinician_event,
            physiology_events=events,
            clinician_meta_events=clinician_meta_events,
            clinician_declared_code=clinician_declared_code,
        )

        if self.current_mode == SimMode.CODE:
            if exit_requested and rosc_now:
                self.current_mode = SimMode.URGENT
                self.downgrade_counter = 0
            elif code_downgrade_allowed:
                self.current_mode = self._code_exit_target(code_downgrade_target)
                self.downgrade_counter = 0
            return self.current_mode

        current_priority = SIM_MODE_PRIORITY[self.current_mode]
        target_priority = SIM_MODE_PRIORITY[target_mode]

        if target_priority > current_priority:
            self.current_mode = target_mode
            self.downgrade_counter = 0
            return self.current_mode

        if target_mode == self.current_mode:
            self.downgrade_counter = 0
            return self.current_mode

        self.downgrade_counter += 1
        if self.downgrade_counter >= self.downgrade_required_rounds:
            self.current_mode = target_mode
            self.downgrade_counter = 0

        return self.current_mode

    @staticmethod
    def _code_exit_target(target: SimMode | str | None) -> SimMode:
        if target is None:
            return SimMode.URGENT
        mode = _coerce_mode(target)
        if mode == SimMode.CODE:
            return SimMode.URGENT
        return mode


def get_round_dt_s(mode: SimMode | str) -> int:
    mode = _coerce_mode(mode)
    if mode == SimMode.CODE:
        return 5
    if mode == SimMode.URGENT:
        return 10
    return 30


def _coerce_mode(mode: SimMode | str) -> SimMode:
    if isinstance(mode, SimMode):
        return mode
    if isinstance(mode, Enum):
        mode = mode.value
    try:
        return SimMode(mode)
    except ValueError:
        return SimMode[str(mode)]


def _collect_meta_event_names(
    clinician_event: Any | None,
    clinician_meta_events: Iterable[Any] | None,
) -> set[str]:
    names: set[str] = set()
    for event in _iter_events(clinician_event):
        names.add(_meta_event_name(event))
    for event in clinician_meta_events or []:
        names.add(_meta_event_name(event))
    return names


def _iter_events(event_or_events: Any | None) -> Iterable[Any]:
    if event_or_events is None:
        return []
    if isinstance(event_or_events, (str, bytes, Enum, Mapping)):
        return [event_or_events]
    if isinstance(event_or_events, Iterable):
        return event_or_events
    return [event_or_events]


def _meta_event_name(event: Any) -> str:
    if isinstance(event, Mapping):
        for key in ("action_type", "speech_act", "event", "kind", "type", "name"):
            if key in event:
                return _normalize_token(event[key])
        return ""
    if isinstance(event, Enum):
        return _normalize_token(event.value)
    for attr in ("action_type", "speech_act", "event", "kind", "type", "name"):
        if hasattr(event, attr):
            return _normalize_token(getattr(event, attr))
    return _normalize_token(event)


def _event_kind(event: Any) -> str:
    if isinstance(event, Mapping):
        return _normalize_token(
            event.get("kind", event.get("event_type", event.get("type", "")))
        )
    for attr in ("kind", "event_type", "type"):
        if hasattr(event, attr):
            return _normalize_token(getattr(event, attr))
    return ""


def _event_payload(event: Any) -> Mapping[str, Any]:
    payload: Any = {}
    if isinstance(event, Mapping):
        payload = event.get("payload", {})
    elif hasattr(event, "payload"):
        payload = getattr(event, "payload")
    if isinstance(payload, Mapping):
        return payload
    return {}


def _event_suggests_code(event: Any) -> bool:
    kind = _event_kind(event)
    if kind in _CODE_PHYSIOLOGY_EVENTS:
        return True
    payload = _event_payload(event)
    return any(
        _is_arrest_rhythm(payload.get(key))
        for key in ("rhythm", "to", "new_rhythm", "after_rhythm")
    )


def _event_suggests_rosc(event: Any) -> bool:
    return _event_kind(event) in _ROSC_EVENTS


def _event_suggests_urgent(event: Any) -> bool:
    return _event_kind(event) in _URGENT_PHYSIOLOGY_EVENTS


def _is_arrest_rhythm(rhythm: Any | None) -> bool:
    return rhythm is not None and _normalize_token(rhythm) in _ARREST_RHYTHMS


def _normalize_token(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _vital_number(
    vitals: Mapping[str, Any] | None,
    *keys: str,
) -> float | None:
    if not vitals:
        return None

    exact_keys = set(keys)
    lower_keys = {key.lower() for key in keys}
    for key, value in vitals.items():
        if key in exact_keys or str(key).lower() in lower_keys:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


__all__ = [
    "ClinicianMetaEvent",
    "DECLARE_CODE",
    "EXIT_CODE",
    "SIM_MODE_PRIORITY",
    "SimMode",
    "SimModeManager",
    "determine_target_mode",
    "get_round_dt_s",
]
