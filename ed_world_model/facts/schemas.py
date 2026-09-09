"""Structured disclosed-fact schemas for known_facts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FactModel(BaseModel):
    """Strict base model for extracted known-fact records."""

    model_config = ConfigDict(extra="forbid")


class KnownSymptom(FactModel):
    name: str
    status: Literal["present", "absent", "uncertain"]
    onset: str | None = None
    severity: str | None = None
    source_texts: list[str] = Field(default_factory=list)


class KnownHistory(FactModel):
    item: str
    status: Literal["present", "absent", "uncertain"]
    source_texts: list[str] = Field(default_factory=list)


class KnownAllergy(FactModel):
    substance: str | None = None
    status: Literal["present", "absent", "uncertain"]
    reaction: str | None = None
    source_texts: list[str] = Field(default_factory=list)


class KnownMedication(FactModel):
    name: str | None = None
    status: Literal["current", "not_taking", "unknown"]
    source_texts: list[str] = Field(default_factory=list)


class FactExtractionResult(FactModel):
    symptoms: list[KnownSymptom] = Field(default_factory=list)
    history: list[KnownHistory] = Field(default_factory=list)
    allergies: list[KnownAllergy] = Field(default_factory=list)
    medications: list[KnownMedication] = Field(default_factory=list)
    ignored: bool = False
    reason: str | None = None


__all__ = [
    "FactExtractionResult",
    "KnownAllergy",
    "KnownHistory",
    "KnownMedication",
    "KnownSymptom",
]
