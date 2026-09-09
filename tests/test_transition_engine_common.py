from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.common import (
    CANONICAL_VITAL_KEYS,
    build_engine_input,
    direction,
    evaluate_dataset,
    evaluate_pair,
    iter_engine_inputs,
    make_baseline_output_from_before,
)


def _case_doc() -> dict:
    return {
        "schema_version": "synthetic_v1",
        "case_id": "case-1",
        "category": "synthetic",
        "scenario_description": "Synthetic transition-pair case.",
        "case_context": {
            "demographics": {"age": 44},
            "supporting_findings": [{"result_summary": "answer leak"}],
        },
        "pairs": [
            {
                "id": "p1",
                "source": {"modifier_text": "HR -> 110"},
                "input": {
                    "before": {
                        "vitals": {
                            "HR": 100,
                            "BP_sys": 120,
                            "BP_dia": 70,
                            "RR": 20,
                            "O2Sat": 94,
                            "T": 37.0,
                        },
                        "features": {"rhythm": "sinus"},
                    },
                    "action": {"raw_text": "synthetic action"},
                },
                "label": {
                    "target": {
                        "vitals": {"HR": 110, "BP_sys": 40, "RR": 25},
                        "features": {"rhythm": "answer"},
                    },
                    "evaluation": {
                        "target_vitals": ["HR"],
                        "target_features": ["rhythm"],
                    },
                },
            },
            {
                "id": "p2",
                "source": {"modifier_text": "RR -> 25, O2 -> 99"},
                "input": {
                    "before": {
                        "vitals": {
                            "HR": 100,
                            "BP_sys": 120,
                            "BP_dia": 70,
                            "RR": 20,
                            "O2Sat": 94,
                            "T": 37.0,
                        }
                    },
                    "action": {"raw_text": "second action"},
                },
                "label": {
                    "target": {"vitals": {"RR": 25, "O2Sat": 99}},
                    "evaluation": {"target_vitals": ["RR", "O2Sat"]},
                },
            },
        ],
    }


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(nested, key) for nested in value.values())
    if isinstance(value, list):
        return any(_contains_key(nested, key) for nested in value)
    return False


def test_build_engine_input_omits_label_source_and_supporting_findings():
    case_doc = _case_doc()
    pair = case_doc["pairs"][0]

    engine_input = build_engine_input(case_doc, pair)

    assert set(engine_input) == {
        "case_id",
        "category",
        "scenario_description",
        "case_context",
        "pair_id",
        "before",
        "action",
    }
    assert "label" not in engine_input
    assert "source" not in engine_input
    assert "supporting_findings" not in engine_input["case_context"]
    assert not _contains_key(engine_input, "modifier_text")
    assert case_doc["case_context"]["supporting_findings"] == [
        {"result_summary": "answer leak"}
    ]


def test_iter_engine_inputs_yields_original_pair_and_sanitized_engine_input():
    case_doc = _case_doc()

    rows = list(iter_engine_inputs(case_doc))

    assert rows[0][0] is case_doc["pairs"][0]
    assert rows[0][1]["pair_id"] == "p1"
    assert "label" not in rows[0][1]
    assert "source" not in rows[0][1]
    assert "supporting_findings" not in rows[0][1]["case_context"]


def test_build_engine_input_returns_deep_mutation_isolated_copy():
    case_doc = _case_doc()
    pair = case_doc["pairs"][0]
    engine_input = build_engine_input(case_doc, pair)

    engine_input["case_context"]["demographics"]["age"] = 99
    engine_input["before"]["vitals"]["HR"] = 999
    engine_input["action"]["raw_text"] = "mutated"

    assert case_doc["case_context"]["demographics"]["age"] == 44
    assert pair["input"]["before"]["vitals"]["HR"] == 100
    assert pair["input"]["action"]["raw_text"] == "synthetic action"


def test_make_baseline_output_from_before_fills_all_vital_keys():
    case_doc = _case_doc()
    pair = case_doc["pairs"][0]
    del pair["input"]["before"]["vitals"]["T"]
    engine_input = build_engine_input(case_doc, pair)

    output = make_baseline_output_from_before(engine_input)

    assert tuple(output["prediction"]["vitals"]) == CANONICAL_VITAL_KEYS
    assert output["prediction"]["vitals"]["HR"] == 100
    assert output["prediction"]["vitals"]["T"] is None
    assert output["prediction"]["features"] == {"rhythm": "sinus"}


@pytest.mark.parametrize(
    ("before", "after", "vital", "expected"),
    [
        (100, 105, "HR", "stable"),
        (100, 106, "HR", "up"),
        (100, 94, "HR", "down"),
        (20, 22, "RR", "stable"),
        (37.0, 37.2, "T", "stable"),
        (37.0, 37.21, "T", "up"),
        (None, 110, "HR", None),
        (100, None, "HR", None),
    ],
)
def test_direction_respects_dead_zones(before, after, vital, expected):
    assert direction(before, after, vital) == expected


def test_evaluate_pair_only_evaluates_target_vitals():
    pair = _case_doc()["pairs"][0]
    engine_output = {
        "prediction": {
            "vitals": {
                "HR": 110,
                "BP_sys": 999,
                "RR": 999,
            },
            "features": {"rhythm": "wrong"},
        }
    }

    result = evaluate_pair(pair, engine_output)

    assert result["target_vitals"] == ["HR"]
    assert result["direction_match"] == {"HR": True}
    assert result["direction_accuracy"] == 1.0
    assert result["normalized_l2"] == 0.0


def test_normalized_l2_uses_scales_not_raw_l2():
    pair = _case_doc()["pairs"][0]
    pair["label"]["target"]["vitals"]["HR"] = 110
    engine_output = {"prediction": {"vitals": {"HR": 130}}}

    result = evaluate_pair(pair, engine_output)

    assert result["normalized_l2"] == pytest.approx(2.0)
    assert result["normalized_l2"] != pytest.approx(20.0)


def test_missing_predictions_and_missing_targets_are_excluded_from_normalized_l2():
    pair = _case_doc()["pairs"][0]
    pair["label"] = {
        "target": {"vitals": {"HR": 110, "RR": 24, "T": None}},
        "evaluation": {"target_vitals": ["HR", "RR", "T"]},
    }
    engine_output = {"prediction": {"vitals": {"RR": 29, "T": 38.0}}}

    result = evaluate_pair(pair, engine_output)

    assert result["missing_predictions"] == {"HR": True, "RR": False, "T": False}
    assert result["num_missing_predictions"] == 1
    assert result["num_missing_targets"] == 1
    assert result["normalized_l2"] == pytest.approx(1.0)
    assert result["direction_match"]["HR"] is False
    assert result["direction_match"]["T"] is False


def test_empty_target_vitals_are_unevaluable():
    pair = _case_doc()["pairs"][0]
    pair["label"] = {
        "target": {"vitals": {"HR": 110}},
        "evaluation": {"target_vitals": []},
    }
    engine_output = {"prediction": {"vitals": {"HR": 110}}}

    result = evaluate_pair(pair, engine_output)

    assert result["target_vitals"] == []
    assert result["direction_match"] == {}
    assert result["direction_accuracy"] is None
    assert result["normalized_l2"] is None
    assert result["skipped_vitals"] == []


def test_missing_before_only_affects_direction_not_normalized_l2():
    pair = _case_doc()["pairs"][0]
    pair["input"]["before"]["vitals"]["HR"] = None
    engine_output = {"prediction": {"vitals": {"HR": 130}}}

    result = evaluate_pair(pair, engine_output)

    assert result["missing_predictions"] == {"HR": False}
    assert result["direction_match"] == {"HR": False}
    assert result["normalized_l2"] == pytest.approx(2.0)


def test_non_canonical_target_vital_is_skipped_and_recorded():
    pair = _case_doc()["pairs"][0]
    pair["label"] = {
        "target": {"vitals": {"HR": 110, "PainScore": 3}},
        "evaluation": {"target_vitals": ["HR", "PainScore"]},
    }
    engine_output = {"prediction": {"vitals": {"HR": 110, "PainScore": 4}}}

    result = evaluate_pair(pair, engine_output)

    assert result["skipped_vitals"] == ["PainScore"]
    assert result["direction_match"] == {"HR": True}
    assert result["direction_accuracy"] == 1.0
    assert result["normalized_l2"] == 0.0
    assert result["missing_predictions"] == {"HR": False, "PainScore": False}


def test_evaluate_dataset_aggregates_pair_results():
    case_doc = _case_doc()
    outputs = {
        "p1": {"prediction": {"vitals": {"HR": 110}}},
        "p2": {"prediction": {"vitals": {"RR": 20, "O2Sat": 94}}},
    }

    result = evaluate_dataset(case_doc, outputs)

    assert result["case_id"] == "case-1"
    assert result["num_pairs"] == 2
    assert len(result["pair_results"]) == 2
    assert result["aggregate"]["direction_accuracy_mean"] == pytest.approx(0.5)
    assert result["aggregate"]["normalized_l2_mean"] == pytest.approx(2**0.5 / 2)
    assert result["aggregate"]["normalized_l2_median"] == pytest.approx(2**0.5 / 2)
    assert result["aggregate"]["num_evaluated_pairs"] == 2


def test_evaluate_dataset_missing_engine_output_for_pair_does_not_crash():
    case_doc = _case_doc()
    outputs = {"p1": {"prediction": {"vitals": {"HR": 110}}}}

    result = evaluate_dataset(case_doc, outputs)

    missing_output_result = result["pair_results"][1]
    assert missing_output_result["pair_id"] == "p2"
    assert missing_output_result["missing_predictions"] == {
        "RR": True,
        "O2Sat": True,
    }
    assert missing_output_result["direction_accuracy"] == 0.0
    assert missing_output_result["normalized_l2"] is None


def test_evaluate_dataset_all_unevaluable_pairs_have_empty_aggregates():
    case_doc = _case_doc()
    case_doc["pairs"][0]["label"] = {
        "target": {"vitals": {"HR": 110}},
        "evaluation": {"target_vitals": []},
    }
    case_doc["pairs"][1]["label"] = {
        "target": {"vitals": {"PainScore": 3}},
        "evaluation": {"target_vitals": ["PainScore"]},
    }
    outputs = {
        "p1": {"prediction": {"vitals": {"HR": 110}}},
        "p2": {"prediction": {"vitals": {"PainScore": 4}}},
    }

    result = evaluate_dataset(case_doc, outputs)

    assert result["num_pairs"] == 2
    assert result["aggregate"]["direction_accuracy_mean"] is None
    assert result["aggregate"]["normalized_l2_mean"] is None
    assert result["aggregate"]["normalized_l2_median"] is None
    assert result["aggregate"]["num_evaluated_pairs"] == 0
