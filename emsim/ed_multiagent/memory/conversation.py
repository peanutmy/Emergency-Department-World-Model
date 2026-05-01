"""Dialogue memory that is not a clinical fact source of truth."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """One raw utterance retained for replay and local dialogue context."""

    speaker: str = ""
    target: str | None = None
    content: str = ""
    time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_s = float(self.time_s)
        self.metadata = deepcopy(self.metadata)


@dataclass
class ConversationMemory:
    """Stores transcript text while keeping clinical facts elsewhere."""

    raw_turns: list[ConversationTurn] = field(default_factory=list)
    recent_window: list[ConversationTurn] = field(default_factory=list)
    max_recent_turns: int = 12
    emotional_or_social_summary: str | None = None

    def __post_init__(self) -> None:
        self.max_recent_turns = int(self.max_recent_turns)
        if self.max_recent_turns < 0:
            raise ValueError("max_recent_turns must be non-negative")

        self.raw_turns = [self._coerce_turn(turn) for turn in self.raw_turns]
        if self.recent_window:
            self.recent_window = [
                self._coerce_turn(turn) for turn in self.recent_window
            ]
        else:
            self.recent_window = list(self.raw_turns)
        self._trim_recent_window()

    def add_turn(
        self,
        turn: ConversationTurn | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationTurn:
        if turn is not None and kwargs:
            raise ValueError("pass either turn or keyword fields, not both")
        conversation_turn = self._coerce_turn(turn if turn is not None else kwargs)
        self.raw_turns.append(conversation_turn)
        self.recent_window.append(conversation_turn)
        self._trim_recent_window()
        return conversation_turn

    @staticmethod
    def _coerce_turn(turn: ConversationTurn | dict[str, Any]) -> ConversationTurn:
        if isinstance(turn, ConversationTurn):
            return deepcopy(turn)
        return ConversationTurn(**turn)

    def _trim_recent_window(self) -> None:
        if self.max_recent_turns == 0:
            self.recent_window = []
            return
        self.recent_window = self.recent_window[-self.max_recent_turns :]
