"""Role-specific observation construction for active ED world-model agents."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ed_world_model.orchestration.orchestrator import (
    CLINICIAN,
    NURSE,
    PATIENT,
    RELATIVE,
)
from ed_world_model.state.global_state import Event, GlobalState, Message


DEFAULT_RECENT_MESSAGE_LIMIT = 10
PUBLIC_RECIPIENTS = {None, "all", "public"}
VISIBLE_BEDSIDE_EVENT_TYPES = {
    "bedside_event",
    "nurse_bedside_verbal_slot",
    "nurse_shadow_execution",
}
VISIBLE_PATIENT_STATUS_FEATURE_KEYS = {
    "appearance",
    "distress",
    "mental_status",
    "oxygen_device",
    "oxygen_delivery",
    "respiratory_distress",
    "visible_distress",
    "work_of_breathing",
}


class ObservationBuilder:
    """Build read-only role views for the agents already selected as active."""

    def __init__(self, recent_message_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT) -> None:
        self.recent_message_limit = recent_message_limit

    def build_for(
        self,
        active_agents: Iterable[str],
        global_state: GlobalState,
    ) -> dict[str, dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        for agent in active_agents:
            if agent == CLINICIAN:
                observations[agent] = self.build_clinician_observation(global_state)
            elif agent == NURSE:
                observations[agent] = self.build_nurse_observation(global_state)
            elif agent == PATIENT:
                observations[agent] = self.build_patient_observation(global_state)
            elif agent == RELATIVE:
                observations[agent] = self.build_relative_observation(global_state)
        return observations

    def build_clinician_observation(
        self,
        global_state: GlobalState,
    ) -> dict[str, Any]:
        runtime_state = global_state.runtime_state
        return {
            "patient_state": _dump(global_state.patient_state),
            "known_facts": _dump(global_state.known_facts),
            "available_diagnostic_tests": [
                test.name for test in global_state.truth_state.test_bank
            ],
            "newly_available_results": _dump_list(
                runtime_state.newly_available_results
            ),
            "recent_messages": _dump_list(
                _recent_messages(
                    runtime_state.messages,
                    self.recent_message_limit,
                )
            ),
            "pending_questions": _dump_list(
                _clinician_aware_pending_questions(global_state)
            ),
            "last_turn_events": _dump_list(runtime_state.last_turn_events),
            "demographics": _dump(global_state.truth_state.demographics),
        }

    def build_nurse_observation(
        self,
        global_state: GlobalState,
    ) -> dict[str, Any]:
        runtime_state = global_state.runtime_state
        last_turn_events = runtime_state.last_turn_events
        return {
            "patient_state": _dump(global_state.patient_state),
            "recent_messages": _dump_list(
                _recent_messages(
                    runtime_state.messages,
                    self.recent_message_limit,
                )
            ),
            "newly_available_results": _dump_list(
                runtime_state.newly_available_results
            ),
            "last_turn_events": _dump_list(last_turn_events),
            "last_bedside_event_if_any": _dump_optional(
                _first_visible_bedside_event(last_turn_events, NURSE)
            ),
            "demographics": _dump(global_state.truth_state.demographics),
        }

    def build_patient_observation(
        self,
        global_state: GlobalState,
    ) -> dict[str, Any]:
        runtime_state = global_state.runtime_state
        return {
            "patient_internal_state": _patient_internal_observation(global_state),
            "patient_emotion": _dump(global_state.psych_state.patient_emotion),
            "recent_messages": _dump_list(
                _recent_messages_for_role(
                    runtime_state.messages,
                    PATIENT,
                    self.recent_message_limit,
                )
            ),
            "perceived_bedside_actions": _dump_list(
                _visible_bedside_events(runtime_state.last_turn_events, PATIENT)
            ),
            "communication_ability": _dump(global_state.patient_state.status_flags),
            "demographics": _dump(global_state.truth_state.demographics),
        }

    def build_relative_observation(
        self,
        global_state: GlobalState,
    ) -> dict[str, Any]:
        runtime_state = global_state.runtime_state
        return {
            "relationship_profile": _dump(global_state.agent_profiles.relative),
            "recent_messages": _dump_list(
                _recent_messages_for_role(
                    runtime_state.messages,
                    RELATIVE,
                    self.recent_message_limit,
                )
            ),
            "visible_patient_status": _visible_patient_status(global_state),
            "visible_last_turn_events": _dump_list(
                _visible_bedside_events(runtime_state.last_turn_events, RELATIVE)
            ),
            "family_side_hidden_info": None,
        }


def _recent_messages(messages: list[Message], limit: int) -> list[Message]:
    if limit <= 0:
        return []
    return messages[-limit:]


def _recent_messages_for_role(
    messages: list[Message],
    role: str,
    limit: int,
) -> list[Message]:
    visible_messages = [
        message
        for message in messages
        if (
            message.recipient in PUBLIC_RECIPIENTS
            or message.recipient == role
            or message.speaker == role
        )
    ]
    return _recent_messages(visible_messages, limit)


def _clinician_aware_pending_questions(global_state: GlobalState) -> list[Any]:
    return [
        question
        for question in global_state.runtime_state.pending_questions
        if (
            question.requires_response
            and not question.is_resolved
            and (
                question.source_agent == CLINICIAN
                or question.target_agent == CLINICIAN
            )
        )
    ]


def _patient_internal_observation(global_state: GlobalState) -> dict[str, Any]:
    patient_internal_state = global_state.truth_state.patient_internal_state
    return {
        "chief_complaint": patient_internal_state.chief_complaint,
        "symptoms": list(patient_internal_state.symptoms),
        "hidden_history": list(patient_internal_state.hidden_history),
        "hidden_allergies": list(patient_internal_state.hidden_allergies),
        "hidden_home_medications": list(
            patient_internal_state.hidden_home_medications
        ),
        "disclosure_rules": patient_internal_state.disclosure_rules,
    }


def _visible_patient_status(global_state: GlobalState) -> dict[str, Any]:
    features = _dump(global_state.patient_state.features)
    return {
        "status_flags": _dump(global_state.patient_state.status_flags),
        "visible_features": {
            key: value
            for key, value in features.items()
            if key in VISIBLE_PATIENT_STATUS_FEATURE_KEYS
        },
    }


def _first_visible_bedside_event(
    events: list[Event],
    role: str,
) -> Event | None:
    visible_events = _visible_bedside_events(events, role)
    if not visible_events:
        return None
    return visible_events[0]


def _visible_bedside_events(events: list[Event], role: str) -> list[Event]:
    return [
        event
        for event in events
        if (
            event.type in VISIBLE_BEDSIDE_EVENT_TYPES
            or _event_explicitly_visible_to(event, role)
        )
    ]


def _event_explicitly_visible_to(event: Event, role: str) -> bool:
    visible_to = event.payload.get("visible_to")
    if visible_to is None:
        return False
    if isinstance(visible_to, str):
        return visible_to in {role, "all", "public"}
    if isinstance(visible_to, list):
        return role in visible_to or "all" in visible_to or "public" in visible_to
    return False


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump())
    return _sanitize(value)


def _dump_list(values: Iterable[Any]) -> list[Any]:
    return [_dump(value) for value in values]


def _dump_optional(value: Any | None) -> Any | None:
    if value is None:
        return None
    return _dump(value)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key != "raw_text"
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


__all__ = [
    "DEFAULT_RECENT_MESSAGE_LIMIT",
    "ObservationBuilder",
]
