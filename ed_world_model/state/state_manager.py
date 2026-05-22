"""State mutation helpers for the v1.3.1 ED world-model runtime."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.state.global_state import (
    DiagnosticResult,
    Event,
    Features,
    GlobalState,
    KnownFacts,
    Message,
    PatientEmotion,
    PatientState,
    PendingDiagnosticResult,
    StatusFlags,
    TestBankItem,
    Vitals,
)


class StateManager:
    """Owns and mutates a single GlobalState instance."""

    def __init__(self, state: GlobalState | None = None) -> None:
        self._state = state if state is not None else GlobalState()

    @property
    def state(self) -> GlobalState:
        return self._state

    def add_message(self, message: Message | dict[str, Any]) -> Message:
        message_model = Message.model_validate(message)
        self._state.runtime_state.messages.append(message_model)
        return message_model

    def record_event(self, event: Event | dict[str, Any]) -> Event:
        event_model = Event.model_validate(event)
        self._state.runtime_state.current_turn_events.append(event_model)
        return event_model

    def update_known_facts(
        self,
        *,
        known_history: str | Iterable[str] | None = None,
        known_allergies: str | Iterable[str] | None = None,
        known_medications: str | Iterable[str] | None = None,
        known_symptoms: str | Iterable[str] | None = None,
    ) -> KnownFacts:
        """Append disclosed clinical facts without reading hidden truth_state."""

        self._state.known_facts.known_history = self._with_extended_string_facts(
            self._state.known_facts.known_history,
            known_history,
        )
        self._state.known_facts.known_allergies = self._with_extended_string_facts(
            self._state.known_facts.known_allergies,
            known_allergies,
        )
        self._state.known_facts.known_medications = self._with_extended_string_facts(
            self._state.known_facts.known_medications,
            known_medications,
        )
        self._state.known_facts.known_symptoms = self._with_extended_string_facts(
            self._state.known_facts.known_symptoms,
            known_symptoms,
        )
        return self._state.known_facts

    def create_pending_diagnostic_result(
        self,
        test_name: str,
        current_turn: int | None = None,
    ) -> PendingDiagnosticResult:
        test = self._find_test_bank_item(test_name)
        ordered_at_turn = self._current_turn(current_turn)
        turnaround_turns = (
            test.turnaround_turns
            if test.turnaround_turns is not None
            else DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
        )
        pending_result = PendingDiagnosticResult(
            test_name=test.name,
            ordered_at_turn=ordered_at_turn,
            ready_at_turn=ordered_at_turn + turnaround_turns,
        )

        self._state.runtime_state.pending_diagnostic_results.append(pending_result)
        self.record_event(
            Event(
                type="diagnostic_order_created",
                turn_index=ordered_at_turn,
                payload={
                    "test_name": test.name,
                    "ordered_at_turn": ordered_at_turn,
                    "ready_at_turn": pending_result.ready_at_turn,
                    "turnaround_turns": turnaround_turns,
                },
            )
        )
        return pending_result

    def release_ready_diagnostic_results(
        self,
        current_turn: int | None = None,
    ) -> list[DiagnosticResult]:
        """Release ready diagnostics into durable and per-turn result stores.

        Returned results are also appended to
        `runtime_state.newly_available_results`, so callers should avoid
        double-counting the returned list and that reminder list as independent
        releases.
        """

        release_turn = self._current_turn(current_turn)
        released_results: list[DiagnosticResult] = []
        still_pending: list[PendingDiagnosticResult] = []

        for pending_result in self._state.runtime_state.pending_diagnostic_results:
            if pending_result.ready_at_turn > release_turn:
                still_pending.append(pending_result)
                continue

            test = self._find_test_bank_item(pending_result.test_name)
            diagnostic_result = DiagnosticResult(name=test.name, result=test.result)
            self._state.known_facts.available_results.append(diagnostic_result)
            self._state.runtime_state.newly_available_results.append(diagnostic_result)
            released_results.append(diagnostic_result)
            self.record_event(
                Event(
                    type="diagnostic_release",
                    turn_index=release_turn,
                    payload={
                        "test_name": test.name,
                        "ordered_at_turn": pending_result.ordered_at_turn,
                        "ready_at_turn": pending_result.ready_at_turn,
                        "released_at_turn": release_turn,
                    },
                )
            )

        self._state.runtime_state.pending_diagnostic_results = still_pending
        return released_results

    def apply_physiology_update(
        self,
        patient_state: PatientState | dict[str, Any] | None = None,
        *,
        vitals: Vitals | dict[str, Any] | None = None,
        features: Features | dict[str, Any] | None = None,
        status_flags: StatusFlags | dict[str, Any] | None = None,
        record_event: bool = False,
    ) -> PatientState:
        if patient_state is not None and any(
            value is not None for value in (vitals, features, status_flags)
        ):
            raise ValueError(
                "Provide either patient_state or individual physiology fields, not both."
            )

        updated_fields: list[str] = []
        if patient_state is not None:
            self._state.patient_state = PatientState.model_validate(patient_state)
            updated_fields.append("patient_state")
        else:
            if vitals is not None:
                self._state.patient_state.vitals = Vitals.model_validate(vitals)
                updated_fields.append("vitals")
            if features is not None:
                self._state.patient_state.features = Features.model_validate(features)
                updated_fields.append("features")
            if status_flags is not None:
                self._state.patient_state.status_flags = StatusFlags.model_validate(
                    status_flags
                )
                updated_fields.append("status_flags")

        if not updated_fields:
            raise ValueError("No physiology update was provided.")

        if record_event:
            self.record_event(
                Event(
                    type="physiology_update",
                    turn_index=self._state.runtime_state.turn_index,
                    payload={"updated_fields": updated_fields},
                )
            )

        return self._state.patient_state

    def update_patient_emotion(
        self,
        patient_emotion: PatientEmotion | dict[str, Any] | None = None,
        *,
        label: str | None = None,
        intensity: Literal["low", "medium", "high"] | None = None,
        notes: str | None = None,
        record_event: bool = False,
    ) -> PatientEmotion:
        if patient_emotion is not None and any(
            value is not None for value in (label, intensity, notes)
        ):
            raise ValueError(
                "Provide either patient_emotion or individual emotion fields, not both."
            )

        if patient_emotion is not None:
            emotion_model = PatientEmotion.model_validate(patient_emotion)
        else:
            if label is None and intensity is None and notes is None:
                raise ValueError("No patient emotion update was provided.")
            emotion_data = self._state.psych_state.patient_emotion.model_dump()
            if label is not None:
                emotion_data["label"] = label
            if intensity is not None:
                emotion_data["intensity"] = intensity
            if notes is not None:
                emotion_data["notes"] = notes
            emotion_model = PatientEmotion.model_validate(emotion_data)

        self._state.psych_state.patient_emotion = emotion_model

        if record_event:
            self.record_event(
                Event(
                    type="emotion_update",
                    turn_index=self._state.runtime_state.turn_index,
                    payload={"label": emotion_model.label},
                )
            )

        return emotion_model

    def advance_turn(self) -> int:
        self._state.runtime_state.turn_index += 1
        self._state.runtime_state.last_turn_events = list(
            self._state.runtime_state.current_turn_events
        )
        self._state.runtime_state.current_turn_events = []
        self._state.runtime_state.newly_available_results = []
        return self._state.runtime_state.turn_index

    def _find_test_bank_item(self, test_name: str) -> TestBankItem:
        for test in self._state.truth_state.test_bank:
            if test.name == test_name:
                return test
        available_names = [test.name for test in self._state.truth_state.test_bank]
        raise ValueError(
            f"Unknown diagnostic test_name {test_name!r}. "
            f"Available tests: {available_names}"
        )

    def _current_turn(self, current_turn: int | None) -> int:
        return (
            self._state.runtime_state.turn_index
            if current_turn is None
            else current_turn
        )

    @staticmethod
    def _with_extended_string_facts(
        existing: list[str],
        values: str | Iterable[str] | None,
    ) -> list[str]:
        if values is None:
            return list(existing)
        if isinstance(values, str):
            return [*existing, values]
        return [*existing, *values]


__all__ = ["StateManager"]
