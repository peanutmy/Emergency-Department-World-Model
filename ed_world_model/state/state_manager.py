"""State mutation helpers for the v1.3.1 ED world-model runtime."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.facts.schemas import (
    FactExtractionResult,
    KnownAllergy,
    KnownHistory,
    KnownMedication,
    KnownSymptom,
)
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
    PendingQuestion,
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
        chief_complaint: str | None = None,
        known_history: Any = None,
        known_allergies: Any = None,
        known_medications: Any = None,
        known_symptoms: Any = None,
    ) -> KnownFacts:
        """Append structured disclosed facts without reading hidden truth_state."""

        if chief_complaint is not None:
            self._state.known_facts.chief_complaint = chief_complaint
        result = FactExtractionResult(
            symptoms=self._coerce_fact_values(known_symptoms, KnownSymptom),
            history=self._coerce_fact_values(known_history, KnownHistory),
            allergies=self._coerce_fact_values(known_allergies, KnownAllergy),
            medications=self._coerce_fact_values(known_medications, KnownMedication),
        )
        self.apply_fact_extraction_result(result)
        return self._state.known_facts

    def apply_fact_extraction_result(
        self,
        result: FactExtractionResult | Mapping[str, Any],
    ) -> KnownFacts:
        """Apply extracted patient/relative facts to known_facts."""

        result_model = FactExtractionResult.model_validate(result)
        if result_model.ignored:
            return self._state.known_facts

        for symptom in result_model.symptoms:
            self._upsert_known_symptom(symptom)

        self._state.known_facts.known_history = self._with_merged_fact_sources(
            self._state.known_facts.known_history,
            result_model.history,
            self._history_key,
        )
        self._state.known_facts.known_allergies = self._with_merged_fact_sources(
            self._state.known_facts.known_allergies,
            result_model.allergies,
            self._allergy_key,
        )
        self._state.known_facts.known_medications = self._with_merged_fact_sources(
            self._state.known_facts.known_medications,
            result_model.medications,
            self._medication_key,
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

    def create_pending_question(
        self,
        *,
        source_agent: str,
        target_agent: str,
        question_text: str,
    ) -> PendingQuestion:
        current_turn = self._state.runtime_state.turn_index
        question = PendingQuestion(
            source_agent=source_agent,
            target_agent=target_agent,
            question_text=question_text,
            question_id=f"q{len(self._state.runtime_state.pending_questions) + 1}",
            created_at_turn=current_turn,
        )
        self._state.runtime_state.pending_questions.append(question)
        if target_agent not in self._state.runtime_state.required_response_agents:
            self._state.runtime_state.required_response_agents.append(target_agent)
        self.record_event(
            Event(
                type="pending_question_created",
                turn_index=current_turn,
                payload={
                    "question_id": question.question_id,
                    "source_agent": source_agent,
                    "target_agent": target_agent,
                },
            )
        )
        return question

    def resolve_pending_questions_for_agent(
        self,
        agent: str,
        *,
        created_before_turn: int | None = None,
    ) -> list[PendingQuestion]:
        current_turn = self._state.runtime_state.turn_index
        resolved: list[PendingQuestion] = []
        for question in self._state.runtime_state.pending_questions:
            if question.target_agent != agent or question.is_resolved:
                continue
            if (
                created_before_turn is not None
                and question.created_at_turn is not None
                and question.created_at_turn >= created_before_turn
            ):
                continue
            question.is_resolved = True
            question.resolved_at_turn = current_turn
            resolved.append(question)

        if resolved and not any(
            question.target_agent == agent and not question.is_resolved
            for question in self._state.runtime_state.pending_questions
        ):
            self._state.runtime_state.required_response_agents = [
                required_agent
                for required_agent in self._state.runtime_state.required_response_agents
                if required_agent != agent
            ]
        if resolved:
            self.record_event(
                Event(
                    type="pending_question_resolved",
                    turn_index=current_turn,
                    payload={
                        "agent": agent,
                        "question_ids": [
                            question.question_id for question in resolved
                        ],
                    },
                )
            )
        return resolved

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
            if self._apply_patient_state_update(patient_state):
                updated_fields.append("patient_state")
        else:
            if vitals is not None and self._apply_vitals_update(vitals):
                updated_fields.append("vitals")
            if features is not None and self._apply_features_update(features):
                updated_fields.append("features")
            if status_flags is not None and self._apply_status_flags_update(
                status_flags
            ):
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

    def _apply_patient_state_update(
        self,
        patient_state: PatientState | dict[str, Any],
    ) -> bool:
        updated = False
        for field_name, update_value in self._provided_patient_state_sections(
            patient_state
        ).items():
            if field_name == "vitals":
                updated = self._apply_vitals_update(update_value) or updated
            elif field_name == "features":
                updated = self._apply_features_update(update_value) or updated
            elif field_name == "status_flags":
                updated = self._apply_status_flags_update(update_value) or updated
        return updated

    def _apply_vitals_update(self, update: Vitals | dict[str, Any]) -> bool:
        return self._merge_physiology_section("vitals", Vitals, update)

    def _apply_features_update(self, update: Features | dict[str, Any]) -> bool:
        return self._merge_physiology_section("features", Features, update)

    def _apply_status_flags_update(
        self,
        update: StatusFlags | dict[str, Any],
    ) -> bool:
        return self._merge_physiology_section("status_flags", StatusFlags, update)

    def _merge_physiology_section(
        self,
        field_name: str,
        model_type: type[Vitals] | type[Features] | type[StatusFlags],
        update: Any,
    ) -> bool:
        update_data = self._provided_update_data(update)
        if not update_data:
            return False

        current_model = getattr(self._state.patient_state, field_name)
        merged_data = current_model.model_dump()
        merged_data.update(update_data)
        setattr(
            self._state.patient_state,
            field_name,
            model_type.model_validate(merged_data),
        )
        return True

    @staticmethod
    def _provided_patient_state_sections(
        patient_state: PatientState | dict[str, Any],
    ) -> dict[str, Any]:
        patient_state_model = PatientState.model_validate(patient_state)
        return {
            key: getattr(patient_state_model, key)
            for key in ("vitals", "features", "status_flags")
            if key in patient_state_model.model_fields_set
        }

    @staticmethod
    def _provided_update_data(update: Any) -> dict[str, Any]:
        if isinstance(update, Mapping):
            return dict(update)
        if hasattr(update, "model_dump"):
            return update.model_dump(exclude_unset=True)
        return dict(update)

    def _upsert_known_symptom(self, symptom: KnownSymptom) -> None:
        new_symptom = symptom.model_copy(deep=True)
        new_key = self._normalized_fact_text(new_symptom.name)
        for existing in self._state.known_facts.known_symptoms:
            if self._normalized_fact_text(existing.name) != new_key:
                continue
            if existing.onset is None and new_symptom.onset is not None:
                existing.onset = new_symptom.onset
            if existing.severity is None and new_symptom.severity is not None:
                existing.severity = new_symptom.severity
            if (
                existing.status == "uncertain"
                and new_symptom.status in {"present", "absent"}
            ):
                existing.status = new_symptom.status
            self._append_new_source_texts(
                existing.source_texts,
                new_symptom.source_texts,
            )
            return

        self._state.known_facts.known_symptoms.append(new_symptom)

    @staticmethod
    def _append_new_source_texts(
        existing: list[str],
        new_values: Iterable[str],
    ) -> None:
        for source_text in new_values:
            if source_text not in existing:
                existing.append(source_text)

    @classmethod
    def _with_merged_fact_sources(
        cls,
        existing: list[Any],
        new_values: Iterable[Any],
        key_fn: Callable[[Any], Any],
    ) -> list[Any]:
        updated = [item.model_copy(deep=True) for item in existing]
        for value in new_values:
            value_copy = value.model_copy(deep=True)
            key = key_fn(value_copy)
            match = next((item for item in updated if key_fn(item) == key), None)
            if match is not None:
                cls._append_new_source_texts(
                    match.source_texts,
                    value_copy.source_texts,
                )
                continue
            updated.append(value_copy)
        return updated

    @classmethod
    def _coerce_fact_values(cls, values: Any, model_type: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, str):
            raise TypeError("known_facts fields require structured fact objects.")
        if isinstance(values, model_type):
            return [values]
        if isinstance(values, Mapping):
            return [model_type.model_validate(values)]
        if isinstance(values, Iterable):
            coerced = []
            for value in values:
                if isinstance(value, str):
                    raise TypeError(
                        "known_facts fields require structured fact objects."
                    )
                coerced.append(model_type.model_validate(value))
            return coerced
        return [model_type.model_validate(values)]

    @classmethod
    def _history_key(cls, history: KnownHistory) -> tuple[str, str]:
        return (
            cls._normalized_fact_text(history.item),
            history.status,
        )

    @classmethod
    def _allergy_key(
        cls,
        allergy: KnownAllergy,
    ) -> tuple[str | None, str, str | None]:
        return (
            cls._normalized_optional_fact_text(allergy.substance),
            allergy.status,
            cls._normalized_optional_fact_text(allergy.reaction),
        )

    @classmethod
    def _medication_key(
        cls,
        medication: KnownMedication,
    ) -> tuple[str | None, str]:
        return (
            cls._normalized_optional_fact_text(medication.name),
            medication.status,
        )

    @classmethod
    def _normalized_optional_fact_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._normalized_fact_text(value)

    @staticmethod
    def _normalized_fact_text(value: str) -> str:
        return " ".join(value.replace("-", " ").strip().lower().split())


__all__ = ["StateManager"]
