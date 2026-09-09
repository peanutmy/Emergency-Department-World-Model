"""Fact extraction interfaces for patient and relative disclosures."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ed_world_model.facts.schemas import FactExtractionResult


class FactExtractionError(ValueError):
    """Raised when fact extraction cannot produce valid structured output."""


class FactExtractionContext(BaseModel):
    """Non-hidden context available to a fact extractor."""

    model_config = ConfigDict(extra="forbid")

    recent_messages: list[Any] = Field(default_factory=list)
    known_facts: Any | None = None
    last_question: str | None = None


class FactExtractor(Protocol):
    def extract(
        self,
        content: str,
        speaker: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        """Extract explicitly disclosed facts from one committed utterance."""


class NoopFactExtractor:
    """Extractor implementation that intentionally records no facts."""

    def extract(
        self,
        content: str,
        speaker: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        return FactExtractionResult(
            ignored=True,
            reason="No fact extractor configured.",
        )


class FakeFactExtractor:
    """Queue-backed extractor for deterministic tests.

    It records calls and returns scripted ``FactExtractionResult`` objects or
    raises scripted exceptions. It never calls external services.
    """

    def __init__(
        self,
        results: Iterable[FactExtractionResult | Mapping[str, Any] | Exception]
        | None = None,
        *,
        default_result: FactExtractionResult | Mapping[str, Any] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._default_result = (
            FactExtractionResult(ignored=True, reason="No scripted extraction.")
            if default_result is None
            else default_result
        )
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def extract(
        self,
        content: str,
        speaker: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        self.calls.append(
            {
                "content": content,
                "speaker": speaker,
                "context": deepcopy(context),
            }
        )
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
        else:
            result = self._default_result
        if isinstance(result, Exception):
            raise result
        return FactExtractionResult.model_validate(result)


__all__ = [
    "FactExtractionContext",
    "FactExtractionError",
    "FactExtractor",
    "FakeFactExtractor",
    "NoopFactExtractor",
]
