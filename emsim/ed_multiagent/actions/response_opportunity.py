"""Response opportunity primitives for no-LLM dialogue state tracking."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..memory.discovered import DiscoveredClinicalMemory

_M1C_SUPPORTED_SLOT_KEYS = {
    "allergies",
    "medications",
    "past_medical_history",
    "symptom_history",
    "vitals",
    "vitals_history",
    "exam_findings",
    "test_results",
    "test_name",
    "test_result",
}


class ResponseMode(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    EVADE = "evade"
    EMOTIONAL_REACTION = "emotional_reaction"
    SILENCE = "silence"
    UNABLE = "unable"


class ResponseStatus(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    REFUSED = "refused"
    EVADED = "evaded"
    IGNORED = "ignored"
    UNABLE_TO_ANSWER = "unable_to_answer"
    EXPIRED = "expired"


@dataclass
class ResponseOpportunity:
    """A chance for a role to respond to a question, not a forced answer."""

    opportunity_id: str = ""
    asked_by: str = "clinician"
    addressed_to: str = "patient"
    question_text: str = ""
    question_type: str = "clarification"
    requiredness: str = "expected"
    sensitivity: str = "low"
    urgency: str = "routine"
    expected_slots: list[str] = field(default_factory=list)
    created_at_s: float = 0.0
    status: ResponseStatus = ResponseStatus.OPEN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_slots = deepcopy(self.expected_slots)
        self.created_at_s = float(self.created_at_s)
        self.status = _coerce_response_status(self.status)
        self.metadata = deepcopy(self.metadata)


class ResponseOpportunityQueue:
    """Insertion-ordered queue of open and resolved response opportunities."""

    def __init__(self, opportunities: list[ResponseOpportunity] | None = None):
        self.opportunities: dict[str, ResponseOpportunity] = {}
        self._next_opportunity_number = 1
        for opportunity in opportunities or []:
            self.add(opportunity)

    def add(
        self,
        opportunity: ResponseOpportunity | dict[str, Any],
    ) -> ResponseOpportunity:
        response_opportunity = self._coerce_opportunity(opportunity)
        if not response_opportunity.opportunity_id:
            response_opportunity.opportunity_id = self._make_opportunity_id()
        if response_opportunity.opportunity_id in self.opportunities:
            raise ValueError(
                f"duplicate opportunity_id {response_opportunity.opportunity_id!r}"
            )
        self.opportunities[response_opportunity.opportunity_id] = (
            response_opportunity
        )
        return response_opportunity

    def next_for(self, role: str) -> ResponseOpportunity | None:
        for opportunity in self.opportunities.values():
            if (
                opportunity.addressed_to == role
                and opportunity.status == ResponseStatus.OPEN
            ):
                return opportunity
        return None

    def list_open(self, role: str | None = None) -> list[ResponseOpportunity]:
        return [
            opportunity
            for opportunity in self.opportunities.values()
            if opportunity.status == ResponseStatus.OPEN
            and (role is None or opportunity.addressed_to == role)
        ]

    def mark_resolved(
        self,
        opportunity_id: str,
        status: ResponseStatus | str,
    ) -> ResponseOpportunity:
        opportunity = self.opportunities[opportunity_id]
        opportunity.status = _coerce_response_status(status)
        return opportunity

    def expire_older_than(
        self,
        now_s: float,
        max_age_s: float,
    ) -> list[ResponseOpportunity]:
        now_s = float(now_s)
        max_age_s = float(max_age_s)
        if max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")

        expired: list[ResponseOpportunity] = []
        for opportunity in self.opportunities.values():
            if (
                opportunity.status == ResponseStatus.OPEN
                and now_s - opportunity.created_at_s >= max_age_s
            ):
                opportunity.status = ResponseStatus.EXPIRED
                expired.append(opportunity)
        return expired

    @staticmethod
    def _coerce_opportunity(
        opportunity: ResponseOpportunity | dict[str, Any],
    ) -> ResponseOpportunity:
        if isinstance(opportunity, ResponseOpportunity):
            return deepcopy(opportunity)
        return ResponseOpportunity(**opportunity)

    def _make_opportunity_id(self) -> str:
        opportunity_id = f"opportunity-{self._next_opportunity_number}"
        self._next_opportunity_number += 1
        return opportunity_id


def status_for_response_mode(response_mode: ResponseMode | str) -> ResponseStatus:
    mode = _coerce_response_mode(response_mode)
    return {
        ResponseMode.DIRECT_ANSWER: ResponseStatus.ANSWERED,
        ResponseMode.PARTIAL_ANSWER: ResponseStatus.PARTIALLY_ANSWERED,
        ResponseMode.REFUSE: ResponseStatus.REFUSED,
        ResponseMode.EVADE: ResponseStatus.EVADED,
        ResponseMode.SILENCE: ResponseStatus.IGNORED,
        ResponseMode.UNABLE: ResponseStatus.UNABLE_TO_ANSWER,
        ResponseMode.CLARIFY: ResponseStatus.OPEN,
        ResponseMode.EMOTIONAL_REACTION: ResponseStatus.OPEN,
    }[mode]


def apply_response_to_discovered_memory(
    opportunity: ResponseOpportunity,
    response_mode: ResponseMode | str,
    slots: dict[str, Any] | None,
    discovered_memory: DiscoveredClinicalMemory,
) -> DiscoveredClinicalMemory:
    """Apply already-structured answer slots to clinician-facing memory.

    This helper intentionally does not parse free text or infer missing facts.
    """

    _ = opportunity
    mode = _coerce_response_mode(response_mode)
    if mode not in {ResponseMode.DIRECT_ANSWER, ResponseMode.PARTIAL_ANSWER}:
        return discovered_memory
    if not slots:
        return discovered_memory

    _validate_response_slots(opportunity, slots)

    if "allergies" in slots:
        discovered_memory.discover_allergies(deepcopy(slots["allergies"]))
    if "medications" in slots:
        discovered_memory.discover_medications(deepcopy(slots["medications"]))
    if "past_medical_history" in slots:
        discovered_memory.discover_past_medical_history(
            deepcopy(slots["past_medical_history"])
        )
    if "symptom_history" in slots:
        discovered_memory.update_symptom_history(deepcopy(slots["symptom_history"]))
    if "vitals" in slots:
        discovered_memory.add_vitals(deepcopy(slots["vitals"]))
    if "vitals_history" in slots:
        vitals_history = deepcopy(slots["vitals_history"])
        if isinstance(vitals_history, list):
            for snapshot in vitals_history:
                discovered_memory.add_vitals(snapshot)
        else:
            discovered_memory.add_vitals(vitals_history)
    if "exam_findings" in slots:
        discovered_memory.update_exam_findings(deepcopy(slots["exam_findings"]))
    if "test_results" in slots:
        for test_name, result in deepcopy(slots["test_results"]).items():
            discovered_memory.add_test_result(test_name, result)
    if "test_name" in slots and "test_result" in slots:
        discovered_memory.add_test_result(
            str(slots["test_name"]),
            deepcopy(slots["test_result"]),
        )
    return discovered_memory


def _validate_response_slots(
    opportunity: ResponseOpportunity,
    slots: dict[str, Any],
) -> None:
    slot_keys = set(slots)
    unsupported = slot_keys - _M1C_SUPPORTED_SLOT_KEYS
    if unsupported:
        raise ValueError(
            "unsupported response slot(s): " + ", ".join(sorted(unsupported))
        )

    if not opportunity.expected_slots:
        return

    expected = set(opportunity.expected_slots)
    unexpected = slot_keys - expected
    if unexpected:
        raise ValueError(
            "unexpected response slot(s): "
            + ", ".join(sorted(unexpected))
            + "; expected one of: "
            + ", ".join(sorted(expected))
        )


def _coerce_response_mode(response_mode: ResponseMode | str) -> ResponseMode:
    if isinstance(response_mode, ResponseMode):
        return response_mode
    try:
        return ResponseMode(response_mode)
    except ValueError:
        return ResponseMode[str(response_mode)]


def _coerce_response_status(status: ResponseStatus | str) -> ResponseStatus:
    if isinstance(status, ResponseStatus):
        return status
    try:
        return ResponseStatus(status)
    except ValueError:
        return ResponseStatus[str(status)]
