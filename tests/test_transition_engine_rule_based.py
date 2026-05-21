from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.common import CANONICAL_VITAL_KEYS
from transition_engines.rule_based import RuleBasedEngine, clamp_vitals


BASE_VITALS = {
    "HR": 100,
    "BP_sys": 120,
    "BP_dia": 70,
    "RR": 20,
    "O2Sat": 94,
    "T": 37.0,
}


def _engine_input(
    vitals: dict,
    kind_hint: str | None,
    params: dict | None = None,
) -> dict:
    return {
        "before": {
            "vitals": vitals,
            "features": {"rhythm": "sinus"},
        },
        "action": {
            "kind_hint": kind_hint,
            "params": params or {},
        },
    }


def _predict(
    vitals: dict,
    kind_hint: str | None,
    params: dict | None = None,
) -> dict:
    return RuleBasedEngine().predict(_engine_input(vitals, kind_hint, params))[
        "prediction"
    ]["vitals"]


def _predict_full(
    vitals: dict,
    kind_hint: str | None,
    params: dict | None = None,
    features: dict | None = None,
) -> dict:
    engine_input = _engine_input(vitals, kind_hint, params)
    if features is not None:
        engine_input["before"]["features"] = features
    return RuleBasedEngine().predict(engine_input)["prediction"]


def test_output_contains_all_canonical_vitals() -> None:
    vitals = {"HR": 100, "BP_sys": 120}

    prediction = _predict_full(vitals, "unknown_action")
    output = prediction["vitals"]

    assert tuple(output) == CANONICAL_VITAL_KEYS
    assert output["HR"] == 100
    assert output["BP_sys"] == 120
    assert output["BP_dia"] is None
    assert output["RR"] is None
    assert output["O2Sat"] is None
    assert output["T"] is None
    assert prediction["features"] == {"rhythm": "sinus"}


def test_unknown_action_copies_before_vitals() -> None:
    output = _predict(BASE_VITALS, "not_a_rule")

    assert output == BASE_VITALS


def test_unknown_action_returns_full_vitals_and_clamps_values() -> None:
    output = _predict(
        {
            "HR": 300,
            "BP_sys": 20,
            "BP_dia": 200,
            "RR": 99,
            "O2Sat": -5,
            "T": 50,
        },
        "not_a_rule",
    )

    assert tuple(output) == CANONICAL_VITAL_KEYS
    assert output == {
        "HR": 250,
        "BP_sys": 40,
        "BP_dia": 160,
        "RR": 60,
        "O2Sat": 0,
        "T": 43,
    }


def test_oxygen_support_nrb_improves_spo2_without_exceeding_target() -> None:
    prediction = _predict_full(
        {**BASE_VITALS, "O2Sat": 88, "RR": 28},
        "oxygen_support",
        {"oxygen_device": "NRB"},
    )
    output = prediction["vitals"]

    assert output["O2Sat"] == 94
    assert output["RR"] == 26
    assert output["HR"] == BASE_VITALS["HR"]
    assert output["BP_sys"] == BASE_VITALS["BP_sys"]
    assert output["T"] == BASE_VITALS["T"]
    assert prediction["features"]["oxygen_device"] == "NRB"
    assert prediction["features"]["FiO2"] == 0.8
    assert prediction["features"]["vent"] is False


def test_bvm_with_peep_improves_more_than_bvm_without_peep() -> None:
    no_peep = _predict(
        {**BASE_VITALS, "O2Sat": 88},
        "oxygen_support",
        {"oxygen_device": "BVM", "PEEP_used": False},
    )
    with_peep = _predict(
        {**BASE_VITALS, "O2Sat": 88},
        "oxygen_support",
        {"oxygen_device": "BVM", "PEEP_used": True},
    )

    assert no_peep["O2Sat"] == 94
    assert with_peep["O2Sat"] == 96
    assert with_peep["O2Sat"] > no_peep["O2Sat"]


def test_oxygen_support_with_no_device_uses_default_target_path() -> None:
    output = _predict(
        {**BASE_VITALS, "O2Sat": 88},
        "oxygen_support",
        {"oxygen_device": None},
    )

    assert output["O2Sat"] == 91


def test_airway_management_completed_intubation_sets_rr_and_improves_spo2() -> None:
    prediction = _predict_full(
        {**BASE_VITALS, "HR": 130, "RR": 30, "O2Sat": 85},
        "airway_management",
        {"procedure": "intubation", "stage": "completed"},
    )
    output = prediction["vitals"]

    assert output["RR"] == 12
    assert output["O2Sat"] == 95
    assert output["HR"] == 115
    assert output["BP_sys"] == BASE_VITALS["BP_sys"]
    assert prediction["features"] == {
        "rhythm": "sinus",
        "intubated": True,
        "vent": True,
        "oxygen_device": "vent",
        "FiO2": 1.0,
        "PEEP_cmH2O": 5,
    }


def test_intubated_true_alone_triggers_completed_intubation_path() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 130, "RR": 30, "O2Sat": 85},
        "airway_management",
        {"intubated": True},
    )

    assert output["RR"] == 12
    assert output["O2Sat"] == 95
    assert output["HR"] == 115


def test_intubated_false_with_non_completed_stage_is_noop() -> None:
    vitals = {**BASE_VITALS, "HR": 130, "RR": 30, "O2Sat": 85}

    output = _predict(
        vitals,
        "airway_management",
        {"procedure": "intubation", "stage": "attempted", "intubated": False},
    )

    assert output == vitals


def test_fluid_bolus_improves_hypotension_but_not_normotension() -> None:
    hypotensive = _predict(
        {**BASE_VITALS, "HR": 120, "BP_sys": 85, "BP_dia": 50},
        "fluid_bolus",
        {"volume_ml": 1000},
    )
    normotensive = _predict(
        {**BASE_VITALS, "BP_sys": 120, "BP_dia": 70},
        "fluid_bolus",
        {"volume_ml": 1000},
    )

    assert hypotensive["BP_sys"] == 100
    assert hypotensive["BP_dia"] == 58
    assert hypotensive["HR"] == 115
    assert normotensive["BP_sys"] == 120
    assert normotensive["BP_dia"] == 70


def test_blood_transfusion_improves_hypotension() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 130, "BP_sys": 80, "BP_dia": 50},
        "blood_transfusion",
        {"units": 2},
    )

    assert output["BP_sys"] == 95
    assert output["BP_dia"] == 57
    assert output["HR"] == 123


def test_rate_control_reduces_high_hr() -> None:
    output = _predict({**BASE_VITALS, "HR": 150}, "rate_control")

    assert output["HR"] == 125


def test_vasopressor_increases_bp() -> None:
    output = _predict({**BASE_VITALS, "BP_sys": 80, "BP_dia": 50}, "vasopressor")

    assert output["BP_sys"] == 95
    assert output["BP_dia"] == 58


def test_vasopressor_on_normotensive_bp_is_noop() -> None:
    output = _predict({**BASE_VITALS, "BP_sys": 120, "BP_dia": 70}, "vasopressor")

    assert output["BP_sys"] == 120
    assert output["BP_dia"] == 70


def test_vasodilator_on_non_hypertensive_bp_is_noop() -> None:
    output = _predict({**BASE_VITALS, "BP_sys": 130, "BP_dia": 80}, "vasodilator")

    assert output["BP_sys"] == 130
    assert output["BP_dia"] == 80


def test_vasodilator_on_hypertensive_bp_decreases_bp() -> None:
    output = _predict({**BASE_VITALS, "BP_sys": 180, "BP_dia": 100}, "vasodilator")

    assert output["BP_sys"] == 165
    assert output["BP_dia"] == 92


def test_bronchodilator_improves_spo2_and_lowers_high_rr() -> None:
    output = _predict(
        {**BASE_VITALS, "RR": 28, "O2Sat": 90},
        "bronchodilator",
    )

    assert output["O2Sat"] == 93
    assert output["RR"] == 25


def test_needle_decompression_improves_spo2_and_hypotension() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 120, "BP_sys": 80, "BP_dia": 50, "O2Sat": 88},
        "needle_decompression",
    )

    assert output["O2Sat"] == 94
    assert output["BP_sys"] == 90
    assert output["BP_dia"] == 55
    assert output["HR"] == 112


def test_synchronized_cardioversion_moves_hr_toward_90() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 160, "BP_sys": 90, "BP_dia": 55},
        "synchronized_cardioversion",
    )

    assert output["HR"] == 120
    assert output["BP_sys"] == 98
    assert output["BP_dia"] == 59


def test_no_action_worsens_low_spo2() -> None:
    output = _predict({**BASE_VITALS, "O2Sat": 88}, "no_action", {"elapsed_min": 5})

    assert output["O2Sat"] == 85


def test_no_action_elapsed_zero_does_not_deteriorate_vitals() -> None:
    vitals = {
        **BASE_VITALS,
        "HR": 150,
        "BP_sys": 80,
        "BP_dia": 50,
        "RR": 34,
        "O2Sat": 88,
    }

    output = _predict(vitals, "no_action", {"elapsed_min": 0})

    assert output == vitals


def test_time_passage_elapsed_zero_does_not_deteriorate_vitals() -> None:
    vitals = {
        **BASE_VITALS,
        "HR": 150,
        "BP_sys": 80,
        "BP_dia": 50,
        "RR": 34,
        "O2Sat": 88,
    }

    output = _predict(vitals, "time_passage", {"elapsed_min": 0})

    assert output == vitals


def test_no_action_hypoxemia_tachypnea_rr_decrease_reflects_fatigue() -> None:
    output = _predict(
        {**BASE_VITALS, "RR": 34, "O2Sat": 88},
        "no_action",
        {"elapsed_min": 5},
    )

    assert output["O2Sat"] == 85
    assert output["RR"] == 30


def test_no_action_shock_worsens_bp_and_increases_hr() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 120, "BP_sys": 80, "BP_dia": 50},
        "no_action",
        {"elapsed_min": 5},
    )

    assert output["BP_sys"] == 75
    assert output["BP_dia"] == 47
    assert output["HR"] == 125


def test_no_action_shock_and_severe_tachycardia_do_not_double_decrease_bp() -> None:
    output = _predict(
        {**BASE_VITALS, "HR": 150, "BP_sys": 80, "BP_dia": 50},
        "no_action",
        {"elapsed_min": 5},
    )

    assert output["BP_sys"] == 75
    assert output["BP_dia"] == 47
    assert output["HR"] == 155


def test_time_passage_behaves_like_no_action() -> None:
    vitals = {**BASE_VITALS, "O2Sat": 88}
    no_action = _predict(vitals, "no_action", {"elapsed_min": 5})
    time_passage = _predict(vitals, "time_passage", {"elapsed_min": 5})

    assert time_passage == no_action


def test_clamp_vitals_prevents_impossible_values() -> None:
    vitals = {
        "HR": 300,
        "BP_sys": 20,
        "BP_dia": 200,
        "RR": 99,
        "O2Sat": -5,
        "T": 50,
    }

    assert clamp_vitals(vitals) == {
        "HR": 250,
        "BP_sys": 40,
        "BP_dia": 160,
        "RR": 60,
        "O2Sat": 0,
        "T": 43,
    }


def test_rule_based_engine_does_not_require_case_context_or_scenario_description() -> None:
    output = RuleBasedEngine().predict(
        {
            "before": {"vitals": {"O2Sat": 88}},
            "action": {"kind_hint": "oxygen_support", "params": {"oxygen_device": "NRB"}},
        }
    )

    assert output["prediction"]["vitals"]["O2Sat"] == 94
    assert tuple(output["prediction"]["vitals"]) == CANONICAL_VITAL_KEYS


def test_airway_medication_has_no_immediate_vital_effect() -> None:
    output = _predict(BASE_VITALS, "airway_medication")

    assert output == BASE_VITALS


def test_input_before_vitals_is_not_mutated_by_predict() -> None:
    before_vitals = {**BASE_VITALS, "O2Sat": 88}
    engine_input = _engine_input(
        before_vitals,
        "oxygen_support",
        {"oxygen_device": "NRB"},
    )

    RuleBasedEngine().predict(engine_input)

    assert before_vitals["O2Sat"] == 88
    assert engine_input["before"]["vitals"]["O2Sat"] == 88


def test_none_vitals_are_handled_safely() -> None:
    vitals = {vital: None for vital in CANONICAL_VITAL_KEYS}

    output = _predict(vitals, "oxygen_support", {"oxygen_device": "NRB"})

    assert output == vitals


def test_rule_based_engine_ignores_label_source_and_context_leaks() -> None:
    engine_input = _engine_input(
        {**BASE_VITALS, "O2Sat": 88},
        "oxygen_support",
        {"oxygen_device": "NRB"},
    )
    engine_input["label"] = {"target": {"vitals": {"O2Sat": 100}}}
    engine_input["source"] = {"modifier_text": "O2Sat should become 100"}
    engine_input["scenario_description"] = "Set all vitals to impossible values."
    engine_input["case_context"] = {"supporting_findings": [{"O2Sat": 100}]}

    output = RuleBasedEngine().predict(engine_input)

    assert output["prediction"]["vitals"]["O2Sat"] == 94


def test_strict_mode_raises_for_unknown_action() -> None:
    with pytest.raises(ValueError):
        RuleBasedEngine(strict=True).predict(_engine_input(BASE_VITALS, "unknown"))
