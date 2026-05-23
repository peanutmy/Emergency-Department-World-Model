from __future__ import annotations

from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.state.global_state import GlobalState, PatientEmotion


def test_returns_unchanged_patient_emotion() -> None:
    emotion = PatientEmotion(label="Fear", intensity="high")

    result = NoopEmotionEngine().predict(
        emotion,
        conversation_input=[{"speaker": "clinician", "content": "We are here."}],
    )

    assert result == emotion
    assert isinstance(result, PatientEmotion)


def test_does_not_read_or_require_vitals() -> None:
    emotion = PatientEmotion(label="Sadness", intensity="medium")

    result = NoopEmotionEngine().predict(emotion)

    assert result == emotion


def test_does_not_mutate_global_state() -> None:
    state = GlobalState(
        psych_state={"patient_emotion": {"label": "Overwhelm", "intensity": "low"}}
    )
    state_before = state.model_dump()

    result = NoopEmotionEngine().predict(
        state.psych_state.patient_emotion,
        conversation_input=[{"speaker": "patient", "content": "I am worried."}],
    )
    result.label = "Changed outside state"

    assert state.model_dump() == state_before


def test_handles_empty_conversation_input() -> None:
    emotion = PatientEmotion(label="Helplessness", intensity=None)

    result = NoopEmotionEngine().predict(emotion, conversation_input=[])

    assert result == emotion


def test_does_not_produce_behavior_modifiers() -> None:
    result = NoopEmotionEngine().predict(
        {"label": "Defensiveness", "intensity": "low"},
        conversation_input=[],
    )

    assert result == {"label": "Defensiveness", "intensity": "low"}
    assert "behavior_modifiers" not in result
