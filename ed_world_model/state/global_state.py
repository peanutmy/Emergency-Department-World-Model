"""Pydantic models for the v1.3.1 ED world-model GlobalState."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.facts.schemas import (
    KnownAllergy,
    KnownHistory,
    KnownMedication,
    KnownSymptom,
)

DEFAULT_MAX_TURNS = 50


class StateModel(BaseModel):
    """Shared base model for strict GlobalState schema boundaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Demographics(StateModel):
    name: str | None = None
    age: int | None = None
    sex: str | None = None
    weight_kg: float | None = None


class AgentProfile(StateModel):
    """Stable scenario-level role configuration, not runtime memory."""

    role: str
    name: str | None = None
    traits: dict[str, str | int | float | bool | list[str] | None] = Field(
        default_factory=dict
    )


class AgentProfiles(StateModel):
    clinician: AgentProfile = Field(
        default_factory=lambda: AgentProfile(role="clinician")
    )
    nurse: AgentProfile = Field(default_factory=lambda: AgentProfile(role="nurse"))
    patient: AgentProfile = Field(default_factory=lambda: AgentProfile(role="patient"))
    relative: AgentProfile = Field(
        default_factory=lambda: AgentProfile(role="relative")
    )


class PatientInternalState(StateModel):
    """Stable hidden patient-side truth used for patient dialogue context.

    Symptoms here are static dialogue facts initialized from scenario text. The
    v1.3.1 physiology and emotion engines do not update this model.
    `disclosure_rules` is free-form prompt guidance, not a deterministic rules
    engine.
    """

    chief_complaint: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    hidden_history: list[str] = Field(default_factory=list)
    hidden_allergies: list[str] = Field(default_factory=list)
    hidden_home_medications: list[str] = Field(default_factory=list)
    disclosure_rules: str | None = None


class TestBankItem(StateModel):
    name: str
    result: str
    turnaround_turns: NonNegativeInt | None = None


class TruthState(StateModel):
    scenario_description: str | None = None
    demographics: Demographics = Field(default_factory=Demographics)
    patient_internal_state: PatientInternalState = Field(
        default_factory=PatientInternalState
    )
    test_bank: list[TestBankItem] = Field(default_factory=list)


class Vitals(StateModel):
    HR: float | None = None
    BP_sys: float | None = None
    BP_dia: float | None = None
    RR: float | None = None
    O2Sat: float | None = None
    T: float | None = None


class Features(StateModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)


class StatusFlags(StateModel):
    is_alive: bool = True
    can_speak: bool = True
    is_conscious: bool = True


class PatientState(StateModel):
    vitals: Vitals = Field(default_factory=Vitals)
    features: Features = Field(default_factory=Features)
    status_flags: StatusFlags = Field(default_factory=StatusFlags)


class DiagnosticResult(StateModel):
    name: str
    result: str


class KnownFacts(StateModel):
    chief_complaint: str | None = None
    known_history: list[KnownHistory] = Field(default_factory=list)
    known_allergies: list[KnownAllergy] = Field(default_factory=list)
    known_medications: list[KnownMedication] = Field(default_factory=list)
    known_symptoms: list[KnownSymptom] = Field(default_factory=list)
    available_results: list[DiagnosticResult] = Field(default_factory=list)


EmotionLevel = Literal["absent", "present", "extreme"]


class AgentEmotion(StateModel):
    """Four-dimension ordinal emotion state for an emotional agent.

    Only the patient and the relative carry emotion in the world model; the
    clinician and nurse do not. This mirrors ``emotion_engine.EmotionState``
    (identical dimensions and levels) so the engine output maps onto it without
    a lossy conversion. ``"absent"`` means the emotion is genuinely not present,
    ``"present"`` means active but non-crisis, and ``"extreme"`` is terminal
    crisis level. ``notes`` records why the state last moved (engine reasoning),
    for observability.
    """

    fear: EmotionLevel = "absent"
    sadness: EmotionLevel = "absent"
    overwhelm: EmotionLevel = "absent"
    shame: EmotionLevel = "absent"
    notes: str | None = None

    def summary(self) -> dict[str, str]:
        """Return the four emotion levels (excluding notes)."""
        return {
            "fear": self.fear,
            "sadness": self.sadness,
            "overwhelm": self.overwhelm,
            "shame": self.shame,
        }


class PsychState(StateModel):
    patient_emotion: AgentEmotion = Field(default_factory=AgentEmotion)
    relative_emotion: AgentEmotion = Field(default_factory=AgentEmotion)


class Message(StateModel):
    speaker: str
    content: str
    turn_index: NonNegativeInt | None = None
    recipient: str | None = None


class Event(StateModel):
    """Simulator/system event for trajectory and observation context.

    Event payloads are event-specific metadata for logging, observation, and
    debugging only. They are not durable clinical state.
    """

    type: str
    turn_index: NonNegativeInt | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PendingQuestion(StateModel):
    source_agent: str
    target_agent: str
    question_text: str
    question_id: str | None = None
    created_at_turn: NonNegativeInt | None = None
    requires_response: bool = True
    is_resolved: bool = False
    resolved_at_turn: NonNegativeInt | None = None


class PendingDiagnosticResult(StateModel):
    test_name: str
    ordered_at_turn: NonNegativeInt
    ready_at_turn: NonNegativeInt


class AdministeredTreatment(StateModel):
    """One executed treatment order - the encounter's persistent MAR entry.

    Unlike per-turn events and the recent-message window, this log never
    scrolls out, so agents can always see what has already been given.
    """

    family: str
    kind_hint: str
    drug_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    turn_index: NonNegativeInt | None = None


class RuntimeState(StateModel):
    turn_index: NonNegativeInt = 0
    max_turns: NonNegativeInt = DEFAULT_MAX_TURNS
    messages: list[Message] = Field(default_factory=list)
    pending_questions: list[PendingQuestion] = Field(default_factory=list)
    required_response_agents: list[str] = Field(default_factory=list)
    pending_diagnostic_results: list[PendingDiagnosticResult] = Field(
        default_factory=list
    )
    administered_treatments: list[AdministeredTreatment] = Field(
        default_factory=list
    )
    newly_available_results: list[DiagnosticResult] = Field(default_factory=list)
    last_turn_events: list[Event] = Field(default_factory=list)
    current_turn_events: list[Event] = Field(default_factory=list)
    termination_status: str | None = None


class GlobalState(StateModel):
    agent_profiles: AgentProfiles = Field(default_factory=AgentProfiles)
    truth_state: TruthState = Field(default_factory=TruthState)
    patient_state: PatientState = Field(default_factory=PatientState)
    known_facts: KnownFacts = Field(default_factory=KnownFacts)
    psych_state: PsychState = Field(default_factory=PsychState)
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)


# A child younger than this cannot hold a conversation: the patient never
# speaks, staff dialogue goes to the bedside relative (parent/guardian), and
# only the relative's emotion is tracked.
PREVERBAL_AGE_YEARS = 3


def patient_is_preverbal(state: GlobalState) -> bool:
    """True if the patient is too young (under 3 years) to participate in dialogue."""
    age = state.truth_state.demographics.age
    return age is not None and age < PREVERBAL_AGE_YEARS


__all__ = [
    "DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS",
    "DEFAULT_MAX_TURNS",
    "AdministeredTreatment",
    "AgentProfile",
    "AgentProfiles",
    "Demographics",
    "DiagnosticResult",
    "Event",
    "Features",
    "GlobalState",
    "KnownAllergy",
    "KnownFacts",
    "KnownHistory",
    "KnownMedication",
    "KnownSymptom",
    "AgentEmotion",
    "EmotionLevel",
    "Message",
    "PatientInternalState",
    "PatientState",
    "PendingDiagnosticResult",
    "PREVERBAL_AGE_YEARS",
    "patient_is_preverbal",
    "PendingQuestion",
    "PsychState",
    "RuntimeState",
    "StatusFlags",
    "TestBankItem",
    "TruthState",
    "Vitals",
]
