"""Adapter wiring the notebook emotion engine into the world-model TurnLoop.

``TurnLoop`` calls ``emotion_engine.predict(current_patient_emotion, ...)`` once
per turn whenever verbal messages were committed. This adapter bridges that seam
to :mod:`ed_world_model.emotion_engine`:

- on the **first** ``predict`` call, if a ``scenario`` was supplied, it seeds the
  opening four-emotion state from scenario + vitals + patient traits via
  ``init_emotion`` (otherwise it starts from the passed-in emotion),
- maps the world-model :class:`PatientEmotion` (a list of ``{label, intensity}``
  entries) to the engine's :class:`EmotionState` (four named fields) and back,
- runs ``step_emotion`` for each utterance spoken by *another* agent (the
  engine's self-utterance guard means the patient's own words never move its
  emotion),
- fails safe: any engine/LLM error leaves the patient emotion unchanged so a bad
  turn can never corrupt simulation state.

Use it in place of ``NoopEmotionEngine``::

    from ed_world_model.adapters.emotion_engine_adapter import EmotionEngineAdapter
    runner = IntegrationRunner(scenario, emotion_engine=EmotionEngineAdapter(
        scenario=state.truth_state.scenario_description,
        initial_vitals=state.patient_state.vitals.model_dump(),
    ))
"""
from __future__ import annotations

from typing import Any

from ed_world_model.emotion_engine import (
    EMOTION_NAMES,
    EmotionState,
    EmotionTrajectory,
    _call_llm_json,
    init_emotion,
    step_emotion,
)
from ed_world_model.state.global_state import (
    EmotionEntry,
    PatientEmotion,
)


class EmotionEngineAdapter:
    """``predict``-compatible emotion engine backed by the notebook engine.

    Parameters
    ----------
    agent_role:
        The role whose emotion is tracked (the listener). Defaults to
        ``"patient"`` — the only emotion the world model currently stores.
    scenario:
        Scenario description used to seed the opening emotion on the first
        ``predict`` call. When ``None`` the adapter skips init and starts from
        the emotion passed in by the turn loop (the original behaviour).
    initial_vitals:
        Patient baseline vitals handed to ``init_emotion`` alongside the
        scenario. Ignored when ``scenario`` is ``None``.
    llm_call:
        Injected ``complete``-style callable returning parsed JSON, matching
        ``emotion_engine.init_emotion`` / ``step_emotion``. Defaults to the real
        Claude client; pass a fake in tests to avoid network calls.
    """

    def __init__(
        self,
        *,
        agent_role: str = "patient",
        scenario: str | None = None,
        initial_vitals: dict[str, Any] | None = None,
        llm_call=_call_llm_json,
    ) -> None:
        self.agent_role = agent_role
        self.scenario = scenario
        self.initial_vitals = initial_vitals or {}
        self.llm_call = llm_call
        self.last_error: str | None = None
        self.init_rationale: str | None = None
        self._initialized = False

    def predict(
        self,
        current_patient_emotion: Any,
        conversation_input: Any | None = None,
        recent_messages: Any | None = None,
        patient_profile_context: Any | None = None,
    ) -> PatientEmotion:
        self.last_error = None

        if self.scenario and not self._initialized:
            state = self._seed_initial_state(patient_profile_context)
        else:
            state = _patient_emotion_to_state(current_patient_emotion)

        trajectory = EmotionTrajectory(
            agent_role=self.agent_role,
            state=state,
            history=[state],
        )

        clinical_context = self._clinical_context(patient_profile_context)
        # Prior turns' dialogue gives the model the conversational context it
        # needs to score each utterance; the current turn's utterances are
        # appended as we walk through them so later utterances see earlier ones.
        dialogue_so_far = _format_dialogue(_iter_messages(recent_messages))

        # Collect one reasoning line per driving utterance so the caller can see
        # WHY the emotion moved (surfaced via PatientEmotion.notes downstream).
        reasoning_lines: list[str] = []
        for message in _iter_messages(conversation_input):
            speaker = message.get("speaker")
            content = message.get("content")
            if not content or not speaker or speaker == self.agent_role:
                continue
            try:
                trajectory = step_emotion(
                    trajectory,
                    utterance=str(content),
                    speaker_role=str(speaker),
                    clinical_context=clinical_context,
                    dialogue=dialogue_so_far,
                    llm_call=self.llm_call,
                )
            except Exception as exc:  # noqa: BLE001 - fail safe on any engine error
                self.last_error = f"emotion step failed: {exc}"
                break
            if trajectory.last_reasoning:
                reasoning_lines.append(
                    f"[{speaker}: {str(content).strip()}] {trajectory.last_reasoning}"
                )
            dialogue_so_far = _append_dialogue_line(
                dialogue_so_far, speaker, str(content)
            )

        patient_emotion = _state_to_patient_emotion(trajectory.state)
        if reasoning_lines:
            patient_emotion.notes = "\n".join(reasoning_lines)
        return patient_emotion

    def _clinical_context(self, patient_profile_context: Any) -> str:
        """Describe who the listener is, so the step prompt can constrain it.

        Combines the scenario description (which conveys the patient's level of
        consciousness and presentation) with any persona/traits from the profile
        context. Without this, the engine scores every utterance as if spoken to
        a fully alert, fully comprehending adult.
        """

        parts: list[str] = []
        if self.scenario:
            parts.append(f"Scenario: {str(self.scenario).strip()}")
        if isinstance(patient_profile_context, dict):
            persona = (
                patient_profile_context.get("persona")
                or patient_profile_context.get("description")
            )
            if persona:
                parts.append(f"Persona: {persona}")
            traits = patient_profile_context.get("traits")
            if traits:
                parts.append(f"Traits: {traits}")
        return "\n".join(parts)

    def _seed_initial_state(self, patient_profile_context: Any) -> EmotionState:
        """Run ``init_emotion`` once to set the opening four-emotion state.

        Reads the patient's traits from the profile context the turn loop
        passes in. Fails safe to an all-``baseline`` state on any error.
        """

        self._initialized = True
        traits: dict[str, Any] = {}
        if isinstance(patient_profile_context, dict):
            traits = patient_profile_context.get("traits") or {}
        try:
            trajectory = init_emotion(
                agent_role=self.agent_role,
                scenario=str(self.scenario),
                initial_vitals=self.initial_vitals,
                agent_traits=traits,
                llm_call=self.llm_call,
            )
        except Exception as exc:  # noqa: BLE001 - fail safe; start from baseline
            self.last_error = f"emotion init failed: {exc}"
            return EmotionState()
        self.init_rationale = trajectory.init_rationale
        return trajectory.state


def _patient_emotion_to_state(value: Any) -> EmotionState:
    """Convert a world-model PatientEmotion (or dict) to an engine EmotionState."""

    try:
        emotion = (
            value
            if isinstance(value, PatientEmotion)
            else PatientEmotion.model_validate(value)
        )
    except (ValueError, TypeError):
        return EmotionState()

    levels: dict[str, str] = {name: "not at all" for name in EMOTION_NAMES}
    for entry in emotion.emotions:
        name = entry.label.lower()
        if name in levels:
            levels[name] = entry.intensity
    return EmotionState(**levels)


def _state_to_patient_emotion(state: EmotionState) -> PatientEmotion:
    """Convert an engine EmotionState back to a world-model PatientEmotion."""

    emotions = [
        EmotionEntry(label=name.capitalize(), intensity=state.level_of(name))
        for name in EMOTION_NAMES
    ]
    return PatientEmotion(emotions=emotions)


def _format_dialogue(messages: list[dict[str, Any]], *, limit: int = 8) -> str:
    """Render the most recent messages as a readable transcript for the prompt."""

    lines: list[str] = []
    for message in messages[-limit:]:
        speaker = message.get("speaker")
        content = message.get("content")
        if not speaker or not content:
            continue
        lines.append(f"{speaker}: {str(content).strip()}")
    return "\n".join(lines)


def _append_dialogue_line(dialogue: str, speaker: str, content: str) -> str:
    """Append one utterance to a running dialogue digest."""

    line = f"{speaker}: {content.strip()}"
    return f"{dialogue}\n{line}" if dialogue else line


def _iter_messages(conversation_input: Any) -> list[dict[str, Any]]:
    if not conversation_input:
        return []
    messages: list[dict[str, Any]] = []
    for item in conversation_input:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, dict):
            messages.append(item)
    return messages


__all__ = ["EmotionEngineAdapter"]
