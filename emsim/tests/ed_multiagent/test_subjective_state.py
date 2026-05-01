from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import sys

import pytest


EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

import ed_multiagent.observation as observation_api
from ed_multiagent.observation import (
    SpeechCapacity,
    SubjectiveState,
    SymptomTimeline,
    palpitation_present,
    subjective_from_observation,
)
from ed_multiagent.observation.subjective import subjective_from_hidden


FORBIDDEN_SUBJECTIVE_KEYS = {
    "hidden_state",
    "mechanism",
    "pathology",
    "diagnosis",
    "active_drug_effects",
    "CO_index",
    "SVR_index",
    "preload_index",
}


def _vitals(
    *,
    hr: float = 80,
    bp_sys: float = 120,
    rr: float = 16,
    o2_sat: float = 98,
) -> dict:
    return {
        "HR": hr,
        "BP_sys": bp_sys,
        "BP_dia": 70,
        "RR": rr,
        "O2Sat": o2_sat,
        "T": 37.0,
    }


def test_low_o2sat_increases_dyspnea_severity():
    normal = subjective_from_observation(_vitals(o2_sat=98))
    hypoxic = subjective_from_observation(_vitals(o2_sat=86))

    assert hypoxic.dyspnea_severity > normal.dyspnea_severity


def test_high_rr_increases_dyspnea_severity():
    normal = subjective_from_observation(_vitals(rr=16))
    tachypneic = subjective_from_observation(_vitals(rr=34))

    assert tachypneic.dyspnea_severity > normal.dyspnea_severity


@pytest.mark.parametrize(
    ("o2_sat", "expected_capacity"),
    [
        (88, SpeechCapacity.SHORT_PHRASES),
        (82, SpeechCapacity.SINGLE_WORDS),
        (76, SpeechCapacity.UNABLE),
    ],
)
def test_very_low_o2sat_reduces_speech_capacity(
    o2_sat: float,
    expected_capacity: SpeechCapacity,
):
    state = subjective_from_observation(_vitals(o2_sat=o2_sat))

    assert state.speech_capacity == expected_capacity


def test_low_bp_sys_increases_dizziness():
    normal = subjective_from_observation(_vitals(bp_sys=120))
    hypotensive = subjective_from_observation(_vitals(bp_sys=80))

    assert hypotensive.dizziness > normal.dizziness


@pytest.mark.parametrize("rhythm", ["SVT", "VT", "VF"])
def test_tachyarrhythmia_rhythm_produces_palpitation(rhythm: str):
    state = subjective_from_observation(_vitals(), rhythm=rhythm)

    assert state.palpitation is True


def test_high_hr_produces_palpitation():
    state = subjective_from_observation(_vitals(hr=145), rhythm="sinus")

    assert state.palpitation is True


def test_low_consciousness_forces_unable_speech_capacity():
    state = subjective_from_observation(
        _vitals(hr=180),
        rhythm="SVT",
        engine_public_state={"consciousness": 0.2},
    )

    assert state.speech_capacity == SpeechCapacity.UNABLE
    assert state.palpitation is False


def test_moderate_consciousness_impairment_increases_confusion_level():
    normal = subjective_from_observation(
        _vitals(),
        engine_public_state={"consciousness": 1.0},
    )
    impaired = subjective_from_observation(
        _vitals(),
        engine_public_state={"consciousness": 0.65},
    )

    assert impaired.confusion_level > normal.confusion_level


def test_public_consciousness_does_not_suppress_vitals_confusion():
    normal = subjective_from_observation(
        _vitals(),
        engine_public_state={"consciousness": 1.0},
    )
    hypoxic = subjective_from_observation(
        _vitals(o2_sat=76),
        engine_public_state={"consciousness": 1.0},
    )

    assert hypoxic.confusion_level > normal.confusion_level


def test_all_severity_scores_are_clamped_between_zero_and_one():
    state = SubjectiveState(
        dyspnea_severity=10,
        confusion_level=-2,
        palpitation=False,
        dizziness=2,
        pain_distress=3,
        speech_capacity="normal",
        visible_distress=5,
    )

    for key in (
        "dyspnea_severity",
        "confusion_level",
        "dizziness",
        "pain_distress",
        "visible_distress",
    ):
        assert 0.0 <= getattr(state, key) <= 1.0


def test_rendered_severity_scores_are_clamped_between_zero_and_one():
    state = subjective_from_observation(
        _vitals(hr=400, bp_sys=-20, rr=120, o2_sat=-5),
        rhythm="SVT",
        engine_public_state={"consciousness": -1, "pain_score": 50},
    )

    for key in (
        "dyspnea_severity",
        "confusion_level",
        "dizziness",
        "pain_distress",
        "visible_distress",
    ):
        assert 0.0 <= getattr(state, key) <= 1.0


def test_subjective_state_does_not_expose_hidden_state_fields():
    state = subjective_from_observation(
        _vitals(),
        rhythm="SVT",
        engine_public_state={"consciousness": 0.8, "pain_score": 7},
    )

    field_names = {field.name for field in fields(state)}
    serialized_keys = set(asdict(state))

    assert field_names.isdisjoint(FORBIDDEN_SUBJECTIVE_KEYS)
    assert serialized_keys.isdisjoint(FORBIDDEN_SUBJECTIVE_KEYS)


def test_package_api_does_not_reexport_hidden_adapter():
    assert "subjective_from_hidden" not in observation_api.__all__


def test_symptom_timeline_stores_longitudinal_history_separately():
    current = subjective_from_observation(_vitals(o2_sat=88))
    timeline = SymptomTimeline(
        onset_s_before_arrival=1800,
        progression="gradual",
        prior_episodes="Similar episode last winter.",
        patient_description="Short of breath since this morning.",
    )

    assert timeline.onset_s_before_arrival == 1800.0
    assert timeline.progression == "gradual"
    assert "onset_s_before_arrival" not in asdict(current)
    assert "patient_description" not in asdict(current)


def test_symptom_timeline_rejects_negative_onset():
    with pytest.raises(ValueError, match="non-negative"):
        SymptomTimeline(
            onset_s_before_arrival=-1,
            progression="unknown",
            patient_description="Chest pain started before arrival.",
        )


def test_symptom_timeline_rejects_unsupported_progression():
    with pytest.raises(ValueError, match="unsupported symptom progression"):
        SymptomTimeline(
            onset_s_before_arrival=60,
            progression="abruptly_then_fixed",
            patient_description="Chest pain started before arrival.",
        )


def test_chronotropic_drive_above_threshold_produces_palpitation():
    assert palpitation_present(_vitals(), chronotropic_drive=2.0) is True


@pytest.mark.parametrize(
    ("rr", "expected_capacity"),
    [
        (34, SpeechCapacity.SHORT_PHRASES),
        (46, SpeechCapacity.SINGLE_WORDS),
    ],
)
def test_high_rr_reduces_speech_capacity(
    rr: float,
    expected_capacity: SpeechCapacity,
):
    state = subjective_from_observation(_vitals(rr=rr))

    assert state.speech_capacity == expected_capacity


def test_pain_score_increases_pain_distress_and_visible_distress():
    no_pain = subjective_from_observation(
        _vitals(),
        engine_public_state={"pain_score": 0},
    )
    pain = subjective_from_observation(
        _vitals(),
        engine_public_state={"pain_score": 8},
    )

    assert pain.pain_distress >= 0.7
    assert pain.visible_distress > no_pain.visible_distress


def test_lowercase_svt_rhythm_produces_palpitation():
    state = subjective_from_observation(_vitals(), rhythm="svt")

    assert state.palpitation is True


def test_hidden_state_adapter_is_duck_typed_and_patient_facing_only():
    class HiddenLike:
        rhythm = "SVT"
        consciousness = 0.8
        chronotropic_drive = 2.0
        PaO2_effective = 60.0
        shunt_fraction = 0.3
        CO_index = 0.6
        preload_index = 0.7

    state = subjective_from_hidden(HiddenLike(), _vitals(o2_sat=94))

    assert state.dyspnea_severity > 0
    assert state.dizziness > 0
    assert set(asdict(state)).isdisjoint(FORBIDDEN_SUBJECTIVE_KEYS)


def test_hidden_state_adapter_severe_hypoxia_can_force_unable_speech():
    class HiddenLike:
        rhythm = "sinus"
        consciousness = 1.0
        PaO2_effective = 40.0
        shunt_fraction = 0.05
        ventilatory_drive = 1.0

    state = subjective_from_hidden(HiddenLike(), _vitals(o2_sat=96))

    assert state.speech_capacity == SpeechCapacity.UNABLE
