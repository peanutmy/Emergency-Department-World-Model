"""Patient-facing subjective state rendering.

The primary renderer accepts portable observation inputs only: observable
vitals, optional rhythm, and optional public engine state such as
consciousness. Rule-engine ``HiddenState`` support is kept behind the
duck-typed compatibility adapter at the bottom of this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


Progression = Literal["sudden", "gradual", "waxing_waning", "unknown"]


# MVP heuristic thresholds for deterministic simulation behavior.
# These are not validated clinical decision rules.
O2SAT_DYSPNEA_NORMAL = 95.0
O2SAT_DYSPNEA_CRITICAL = 82.0
O2SAT_CONFUSION_NORMAL = 88.0
O2SAT_CONFUSION_CRITICAL = 72.0
O2SAT_SPEECH_SHORT_PHRASES = 90.0
O2SAT_SPEECH_SINGLE_WORDS = 84.0
O2SAT_SPEECH_UNABLE = 78.0

RR_DYSPNEA_NORMAL = 20.0
RR_DYSPNEA_CRITICAL = 40.0
RR_SPEECH_SHORT_PHRASES = 32.0
RR_SPEECH_SINGLE_WORDS = 44.0

BP_SYS_DIZZINESS_NORMAL = 100.0
BP_SYS_DIZZINESS_CRITICAL = 70.0
BP_SYS_CONFUSION_NORMAL = 85.0
BP_SYS_CONFUSION_CRITICAL = 55.0

HR_PALPITATION_THRESHOLD = 130.0
HR_DIZZINESS_NORMAL = 120.0
HR_DIZZINESS_CRITICAL = 190.0

CHRONOTROPIC_DRIVE_PALPITATION_THRESHOLD = 1.5
CONSCIOUSNESS_UNABLE_THRESHOLD = 0.35
CONSCIOUSNESS_SINGLE_WORDS_THRESHOLD = 0.55

DYSPNEA_SPEECH_UNABLE_THRESHOLD = 0.98
DYSPNEA_SPEECH_SINGLE_WORDS_THRESHOLD = 0.85
DYSPNEA_SPEECH_SHORT_PHRASES_THRESHOLD = 0.55

VITALS_CONFUSION_HYPOXIA_WEIGHT = 0.35
VITALS_CONFUSION_HYPOTENSION_WEIGHT = 0.25
VITALS_DIZZINESS_TACHYCARDIA_WEIGHT = 0.3

VISIBLE_DISTRESS_DYSPNEA_WEIGHT = 0.9
VISIBLE_DISTRESS_CONFUSION_WEIGHT = 0.5
VISIBLE_DISTRESS_DIZZINESS_WEIGHT = 0.75
VISIBLE_DISTRESS_PAIN_WEIGHT = 0.9

PAIN_SCORE_MAX = 10.0

# Adapter-only heuristics for current rule-based EMSim hidden-state fields.
HIDDEN_PAO2_DYSPNEA_NORMAL = 80.0
HIDDEN_PAO2_DYSPNEA_CRITICAL = 55.0
HIDDEN_SHUNT_DYSPNEA_NORMAL = 0.10
HIDDEN_SHUNT_DYSPNEA_CRITICAL = 0.35
HIDDEN_VENTILATORY_DRIVE_DYSPNEA_NORMAL = 1.2
HIDDEN_VENTILATORY_DRIVE_DYSPNEA_CRITICAL = 2.2
HIDDEN_CO_INDEX_DIZZINESS_NORMAL = 0.85
HIDDEN_CO_INDEX_DIZZINESS_CRITICAL = 0.45
HIDDEN_PRELOAD_DIZZINESS_NORMAL = 0.80
HIDDEN_PRELOAD_DIZZINESS_CRITICAL = 0.40


class SpeechCapacity(str, Enum):
    NORMAL = "normal"
    SHORT_PHRASES = "short_phrases"
    SINGLE_WORDS = "single_words"
    UNABLE = "unable"

    def __str__(self) -> str:
        return self.value


@dataclass
class SubjectiveState:
    """Current patient-facing experience snapshot.

    This intentionally contains no raw vitals, diagnosis, pathology, or hidden
    physiology fields.
    """

    dyspnea_severity: float
    confusion_level: float
    palpitation: bool
    dizziness: float
    pain_distress: float
    speech_capacity: SpeechCapacity
    visible_distress: float

    def __post_init__(self) -> None:
        self.dyspnea_severity = _clamp01(self.dyspnea_severity)
        self.confusion_level = _clamp01(self.confusion_level)
        self.palpitation = bool(self.palpitation)
        self.dizziness = _clamp01(self.dizziness)
        self.pain_distress = _clamp01(self.pain_distress)
        self.speech_capacity = _coerce_speech_capacity(self.speech_capacity)
        self.visible_distress = _clamp01(self.visible_distress)


@dataclass
class SymptomTimeline:
    """Longitudinal symptom history, separate from current subjective state."""

    onset_s_before_arrival: float | None
    progression: Progression = "unknown"
    prior_episodes: str | None = None
    patient_description: str = ""

    def __post_init__(self) -> None:
        if self.onset_s_before_arrival is not None:
            onset = float(self.onset_s_before_arrival)
            if onset < 0:
                raise ValueError("onset_s_before_arrival must be non-negative")
            self.onset_s_before_arrival = onset
        if self.progression not in _PROGRESSION_VALUES:
            raise ValueError(f"unsupported symptom progression: {self.progression!r}")
        if self.prior_episodes is not None:
            self.prior_episodes = str(self.prior_episodes)
        self.patient_description = str(self.patient_description)


def subjective_from_observation(
    vitals: Mapping[str, Any],
    rhythm: str | None = None,
    engine_public_state: Mapping[str, Any] | None = None,
) -> SubjectiveState:
    """Render patient-facing subjective state from portable engine outputs."""

    public_state = engine_public_state or {}
    consciousness = _public_consciousness(public_state)
    resolved_rhythm = rhythm or _first_string(
        public_state,
        ("rhythm", "cardiac_rhythm"),
    )

    dyspnea = dyspnea_severity(vitals)
    confusion = confusion_level(vitals, consciousness=consciousness)
    dizziness_score = dizziness_severity(vitals)
    pain_score = pain_distress_from_public_state(public_state)
    capacity = speech_capacity(
        vitals,
        consciousness=consciousness,
        dyspnea=dyspnea,
    )
    palpitations = palpitation_present(
        vitals,
        rhythm=resolved_rhythm,
        speech_capacity=capacity,
    )
    distress = visible_distress(
        dyspnea=dyspnea,
        confusion=confusion,
        dizziness=dizziness_score,
        pain_distress=pain_score,
    )

    return SubjectiveState(
        dyspnea_severity=dyspnea,
        confusion_level=confusion,
        palpitation=palpitations,
        dizziness=dizziness_score,
        pain_distress=pain_score,
        speech_capacity=capacity,
        visible_distress=distress,
    )


def dyspnea_severity(vitals: Mapping[str, Any]) -> float:
    """Infer current shortness-of-breath severity from observable vitals."""

    o2_sat = _vital_float(vitals, "O2Sat")
    rr = _vital_float(vitals, "RR")
    hypoxia = _below_score(
        o2_sat,
        normal=O2SAT_DYSPNEA_NORMAL,
        critical=O2SAT_DYSPNEA_CRITICAL,
    )
    tachypnea = _above_score(
        rr,
        normal=RR_DYSPNEA_NORMAL,
        critical=RR_DYSPNEA_CRITICAL,
    )
    return _clamp01(max(hypoxia, tachypnea))


def confusion_level(
    vitals: Mapping[str, Any],
    *,
    consciousness: float | None = None,
) -> float:
    """Infer confusion from public consciousness and observable vitals."""

    vitals_confusion = _vitals_confusion_level(vitals)
    if consciousness is not None:
        consciousness_confusion = _clamp01(1.0 - _clamp01(consciousness))
        return _clamp01(max(consciousness_confusion, vitals_confusion))

    return vitals_confusion


def _vitals_confusion_level(vitals: Mapping[str, Any]) -> float:
    o2_sat = _vital_float(vitals, "O2Sat")
    bp_sys = _vital_float(vitals, "BP_sys")
    hypoxic_component = (
        _below_score(
            o2_sat,
            normal=O2SAT_CONFUSION_NORMAL,
            critical=O2SAT_CONFUSION_CRITICAL,
        )
        * VITALS_CONFUSION_HYPOXIA_WEIGHT
    )
    hypotensive_component = (
        _below_score(
            bp_sys,
            normal=BP_SYS_CONFUSION_NORMAL,
            critical=BP_SYS_CONFUSION_CRITICAL,
        )
        * VITALS_CONFUSION_HYPOTENSION_WEIGHT
    )
    return _clamp01(max(hypoxic_component, hypotensive_component))


def palpitation_present(
    vitals: Mapping[str, Any],
    *,
    rhythm: str | None = None,
    speech_capacity: SpeechCapacity | str | None = None,
    chronotropic_drive: float | None = None,
) -> bool:
    """Return whether the patient can report feeling palpitations."""

    capacity = (
        _coerce_speech_capacity(speech_capacity)
        if speech_capacity is not None
        else None
    )
    if capacity == SpeechCapacity.UNABLE:
        return False

    hr = _vital_float(vitals, "HR")
    normalized_rhythm = _normalize_rhythm(rhythm)
    if normalized_rhythm in {"SVT", "VT", "VF"}:
        return True
    if hr is not None and hr >= HR_PALPITATION_THRESHOLD:
        return True
    if (
        chronotropic_drive is not None
        and chronotropic_drive > CHRONOTROPIC_DRIVE_PALPITATION_THRESHOLD
    ):
        return True
    return False


def dizziness_severity(vitals: Mapping[str, Any]) -> float:
    """Infer dizziness or presyncope from observable perfusion markers."""

    bp_sys = _vital_float(vitals, "BP_sys")
    hr = _vital_float(vitals, "HR")
    hypotension = _below_score(
        bp_sys,
        normal=BP_SYS_DIZZINESS_NORMAL,
        critical=BP_SYS_DIZZINESS_CRITICAL,
    )
    tachycardia = (
        _above_score(
            hr,
            normal=HR_DIZZINESS_NORMAL,
            critical=HR_DIZZINESS_CRITICAL,
        )
        * VITALS_DIZZINESS_TACHYCARDIA_WEIGHT
    )
    return _clamp01(max(hypotension, tachycardia))


def speech_capacity(
    vitals: Mapping[str, Any],
    *,
    consciousness: float | None = None,
    dyspnea: float | None = None,
) -> SpeechCapacity:
    """Deterministically render the patient's physical speech capacity."""

    if consciousness is not None:
        level = _clamp01(consciousness)
        if level <= CONSCIOUSNESS_UNABLE_THRESHOLD:
            return SpeechCapacity.UNABLE
        if level <= CONSCIOUSNESS_SINGLE_WORDS_THRESHOLD:
            return SpeechCapacity.SINGLE_WORDS

    o2_sat = _vital_float(vitals, "O2Sat")
    rr = _vital_float(vitals, "RR")
    if o2_sat is not None:
        if o2_sat <= O2SAT_SPEECH_UNABLE:
            return SpeechCapacity.UNABLE
        if o2_sat <= O2SAT_SPEECH_SINGLE_WORDS:
            return SpeechCapacity.SINGLE_WORDS
        if o2_sat <= O2SAT_SPEECH_SHORT_PHRASES:
            return SpeechCapacity.SHORT_PHRASES

    if rr is not None:
        if rr >= RR_SPEECH_SINGLE_WORDS:
            return SpeechCapacity.SINGLE_WORDS
        if rr >= RR_SPEECH_SHORT_PHRASES:
            return SpeechCapacity.SHORT_PHRASES

    if dyspnea is not None:
        if dyspnea >= DYSPNEA_SPEECH_UNABLE_THRESHOLD:
            return SpeechCapacity.UNABLE
        if dyspnea >= DYSPNEA_SPEECH_SINGLE_WORDS_THRESHOLD:
            return SpeechCapacity.SINGLE_WORDS
        if dyspnea >= DYSPNEA_SPEECH_SHORT_PHRASES_THRESHOLD:
            return SpeechCapacity.SHORT_PHRASES

    return SpeechCapacity.NORMAL


def visible_distress(
    *,
    dyspnea: float,
    confusion: float,
    dizziness: float,
    pain_distress: float,
) -> float:
    """Combine current subjective/observable distress into one visible score."""

    return _clamp01(
        max(
            _clamp01(dyspnea) * VISIBLE_DISTRESS_DYSPNEA_WEIGHT,
            _clamp01(confusion) * VISIBLE_DISTRESS_CONFUSION_WEIGHT,
            _clamp01(dizziness) * VISIBLE_DISTRESS_DIZZINESS_WEIGHT,
            _clamp01(pain_distress) * VISIBLE_DISTRESS_PAIN_WEIGHT,
        )
    )


def pain_distress_from_public_state(
    engine_public_state: Mapping[str, Any] | None,
) -> float:
    """Read optional patient-facing pain fields from a public engine contract."""

    if not engine_public_state:
        return 0.0
    direct = _first_float(
        engine_public_state,
        ("pain_distress", "pain_severity", "pain"),
    )
    if direct is not None:
        return _clamp01(direct if direct <= 1.0 else direct / PAIN_SCORE_MAX)

    score = _first_float(engine_public_state, ("pain_score", "pain_nrs"))
    if score is None:
        return 0.0
    return _clamp01(score / PAIN_SCORE_MAX)


def subjective_from_hidden(h: Any, vitals: Mapping[str, Any]) -> SubjectiveState:
    """Compatibility adapter for the current rule-based EMSim hidden state.

    This function deliberately avoids importing ``rule_engine`` symbols. It
    duck-types fields when available, folds them into subjective scores, and
    still returns only the patient-facing `SubjectiveState` contract.
    It should remain the only M2 renderer path that touches duck-typed engine
    internals.
    """

    public_state: dict[str, Any] = {}
    consciousness = _object_float(h, "consciousness")
    if consciousness is not None:
        public_state["consciousness"] = consciousness

    pain_distress = _object_float(h, "pain_distress")
    if pain_distress is not None:
        public_state["pain_distress"] = pain_distress

    rhythm = _object_string(h, "rhythm")
    base = subjective_from_observation(
        vitals,
        rhythm=rhythm,
        engine_public_state=public_state,
    )

    dyspnea = max(base.dyspnea_severity, _hidden_dyspnea_severity(h))
    dizziness = max(base.dizziness, _hidden_dizziness_severity(h))
    confusion = base.confusion_level
    pain = base.pain_distress
    capacity = speech_capacity(
        vitals,
        consciousness=consciousness,
        dyspnea=dyspnea,
    )
    palpitations = palpitation_present(
        vitals,
        rhythm=rhythm,
        speech_capacity=capacity,
        chronotropic_drive=_object_float(h, "chronotropic_drive"),
    )

    return SubjectiveState(
        dyspnea_severity=dyspnea,
        confusion_level=confusion,
        palpitation=palpitations,
        dizziness=dizziness,
        pain_distress=pain,
        speech_capacity=capacity,
        visible_distress=visible_distress(
            dyspnea=dyspnea,
            confusion=confusion,
            dizziness=dizziness,
            pain_distress=pain,
        ),
    )


def _hidden_dyspnea_severity(h: Any) -> float:
    pa_o2 = _object_float(h, "PaO2_effective")
    shunt = _object_float(h, "shunt_fraction")
    ventilatory_drive = _object_float(h, "ventilatory_drive")
    return _clamp01(
        max(
            _below_score(
                pa_o2,
                normal=HIDDEN_PAO2_DYSPNEA_NORMAL,
                critical=HIDDEN_PAO2_DYSPNEA_CRITICAL,
            ),
            _above_score(
                shunt,
                normal=HIDDEN_SHUNT_DYSPNEA_NORMAL,
                critical=HIDDEN_SHUNT_DYSPNEA_CRITICAL,
            ),
            _above_score(
                ventilatory_drive,
                normal=HIDDEN_VENTILATORY_DRIVE_DYSPNEA_NORMAL,
                critical=HIDDEN_VENTILATORY_DRIVE_DYSPNEA_CRITICAL,
            ),
        )
    )


def _hidden_dizziness_severity(h: Any) -> float:
    co_index = _object_float(h, "CO_index")
    preload_index = _object_float(h, "preload_index")
    return _clamp01(
        max(
            _below_score(
                co_index,
                normal=HIDDEN_CO_INDEX_DIZZINESS_NORMAL,
                critical=HIDDEN_CO_INDEX_DIZZINESS_CRITICAL,
            ),
            _below_score(
                preload_index,
                normal=HIDDEN_PRELOAD_DIZZINESS_NORMAL,
                critical=HIDDEN_PRELOAD_DIZZINESS_CRITICAL,
            ),
        )
    )


def _public_consciousness(
    engine_public_state: Mapping[str, Any],
) -> float | None:
    value = _first_float(
        engine_public_state,
        ("consciousness", "consciousness_level", "alertness"),
    )
    return None if value is None else _clamp01(value)


def _coerce_speech_capacity(value: SpeechCapacity | str) -> SpeechCapacity:
    if isinstance(value, SpeechCapacity):
        return value
    try:
        return SpeechCapacity(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unsupported speech capacity: {value!r}") from exc


def _vital_float(vitals: Mapping[str, Any], key: str) -> float | None:
    return _float_or_none(vitals.get(key))


def _first_float(
    values: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        if key in values:
            value = _float_or_none(values[key])
            if value is not None:
                return value
    return None


def _first_string(
    values: Mapping[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return str(value)
    return None


def _object_float(value: Any, name: str) -> float | None:
    if isinstance(value, Mapping):
        return _float_or_none(value.get(name))
    return _float_or_none(getattr(value, name, None))


def _object_string(value: Any, name: str) -> str | None:
    raw = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    return None if raw is None else str(raw)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rhythm(rhythm: str | None) -> str | None:
    if rhythm is None:
        return None
    normalized = str(rhythm).strip()
    upper = normalized.upper()
    if upper in {"SVT", "VT", "VF", "PEA"}:
        return upper
    return normalized.lower()


def _below_score(
    value: float | None,
    *,
    normal: float,
    critical: float,
) -> float:
    if value is None:
        return 0.0
    if value >= normal:
        return 0.0
    if value <= critical:
        return 1.0
    return _clamp01((normal - value) / (normal - critical))


def _above_score(
    value: float | None,
    *,
    normal: float,
    critical: float,
) -> float:
    if value is None:
        return 0.0
    if value <= normal:
        return 0.0
    if value >= critical:
        return 1.0
    return _clamp01((value - normal) / (critical - normal))


def _clamp01(value: float | int | None) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(1.0, numeric))


_PROGRESSION_VALUES = {"sudden", "gradual", "waxing_waning", "unknown"}


__all__ = [
    "Progression",
    "SpeechCapacity",
    "SubjectiveState",
    "SymptomTimeline",
    "confusion_level",
    "dizziness_severity",
    "dyspnea_severity",
    "pain_distress_from_public_state",
    "palpitation_present",
    "speech_capacity",
    "subjective_from_hidden",
    "subjective_from_observation",
    "visible_distress",
]
