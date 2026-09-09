"""No-op patient emotion engine for v1.3.1."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class NoopEmotionEngine:
    """Return the current patient emotion unchanged."""

    def predict(
        self,
        current_patient_emotion: Any,
        conversation_input: Any | None = None,
        recent_messages: Any | None = None,
        patient_profile_context: Any | None = None,
    ) -> Any:
        del conversation_input, recent_messages, patient_profile_context
        if hasattr(current_patient_emotion, "model_copy"):
            return current_patient_emotion.model_copy(deep=True)
        return deepcopy(current_patient_emotion)


__all__ = ["NoopEmotionEngine"]
