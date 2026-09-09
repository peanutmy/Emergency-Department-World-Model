from __future__ import annotations

from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.state.global_state import AgentEmotion, GlobalState


def test_returns_unchanged_patient_emotion() -> None:
    emotion = AgentEmotion(fear="present", sadness="present")

    result = NoopEmotionEngine().predict(
        emotion,
        conversation_input=[{"speaker": "clinician", "content": "We are here."}],
    )

    assert result == emotion
    assert isinstance(result, AgentEmotion)


def test_does_not_read_or_require_vitals() -> None:
    emotion = AgentEmotion(sadness="present")

    result = NoopEmotionEngine().predict(emotion)

    assert result == emotion


def test_does_not_mutate_global_state() -> None:
    state = GlobalState(
        psych_state={"patient_emotion": {"overwhelm": "present"}}
    )
    state_before = state.model_dump()

    result = NoopEmotionEngine().predict(
        state.psych_state.patient_emotion,
        conversation_input=[{"speaker": "patient", "content": "I am worried."}],
    )
    result.notes = "Changed outside state"

    assert state.model_dump() == state_before


def test_handles_empty_conversation_input() -> None:
    emotion = AgentEmotion(fear="present")

    result = NoopEmotionEngine().predict(emotion, conversation_input=[])

    assert result == emotion


def test_does_not_produce_behavior_modifiers() -> None:
    result = NoopEmotionEngine().predict(
        {"shame": "present"},
        conversation_input=[],
    )

    assert result == {"shame": "present"}
    assert "behavior_modifiers" not in result
