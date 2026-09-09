from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.compare_three_eval_results import (
    PAIR_RESULTS_FILENAME,
    SUMMARY_FILENAME,
    compare_three_eval_results,
)


FIELDNAMES = [
    "case_id",
    "source_file",
    "pair_id",
    "pair_type",
    "kind_hint",
    "target_vitals",
    "direction_accuracy",
    "normalized_l2",
    "predicted_vitals",
    "llm_adjustments",
    "llm_reasoning",
    "llm_api_error",
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in (
                "target_vitals",
                "predicted_vitals",
                "llm_adjustments",
                "llm_reasoning",
            ):
                if key in encoded and not isinstance(encoded[key], str):
                    encoded[key] = json.dumps(encoded[key], sort_keys=True)
            writer.writerow(encoded)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _base_row(pair_id: str, *, kind_hint: str = "oxygen_support") -> dict[str, object]:
    return {
        "case_id": "case-1",
        "source_file": "transitions/case.json",
        "pair_id": pair_id,
        "pair_type": "physiology_response",
        "kind_hint": kind_hint,
        "target_vitals": ["O2Sat"],
        "direction_accuracy": 1.0,
        "normalized_l2": 0.0,
        "predicted_vitals": {"O2Sat": 94},
        "llm_adjustments": {"O2Sat": None},
        "llm_reasoning": {},
        "llm_api_error": "",
    }


def test_compare_three_eval_results_writes_pair_csv_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rule_csv = tmp_path / "rule_based_pair_results.csv"
    hybrid_csv = tmp_path / "hybrid_pair_results.csv"
    pure_llm_csv = tmp_path / "pure_llm_pair_results.csv"
    output_dir = tmp_path / "out"
    _write_rows(
        rule_csv,
        [
            {
                **_base_row("p1"),
                "direction_accuracy": 0.5,
                "normalized_l2": 2.0,
                "predicted_vitals": {"O2Sat": 84},
            },
            {
                **_base_row("p2", kind_hint="no_action"),
                "direction_accuracy": 1.0,
                "normalized_l2": 1.0,
                "predicted_vitals": {"O2Sat": 89},
            },
            {
                **_base_row("p3"),
                "direction_accuracy": 0.75,
                "normalized_l2": 0.5,
                "predicted_vitals": {"O2Sat": 91},
            },
        ],
    )
    _write_rows(
        hybrid_csv,
        [
            {
                **_base_row("p1"),
                "direction_accuracy": 1.0,
                "normalized_l2": 1.0,
                "predicted_vitals": {"O2Sat": 90},
                "llm_adjustments": {"O2Sat": 2},
                "llm_reasoning": {
                    "O2Sat": "Hybrid corrected the rule oxygen response."
                },
            },
            {
                **_base_row("p2", kind_hint="no_action"),
                "direction_accuracy": 1.0,
                "normalized_l2": 1.1,
                "predicted_vitals": {"O2Sat": 89},
                "llm_adjustments": {"O2Sat": None},
            },
            {
                **_base_row("p3"),
                "direction_accuracy": 0.25,
                "normalized_l2": 0.5,
                "predicted_vitals": {"O2Sat": 88},
                "llm_adjustments": {"O2Sat": -3},
                "llm_api_error": "hybrid error",
            },
        ],
    )
    _write_rows(
        pure_llm_csv,
        [
            {
                **_base_row("p1"),
                "direction_accuracy": 1.0,
                "normalized_l2": 0.25,
                "predicted_vitals": {"O2Sat": 93},
                "llm_reasoning": {"O2Sat": "Pure LLM predicted improvement."},
            },
            {
                **_base_row("p2", kind_hint="no_action"),
                "direction_accuracy": 0.0,
                "normalized_l2": 2.0,
                "predicted_vitals": {"O2Sat": 84},
                "llm_api_error": "pure error",
            },
            {
                **_base_row("p3"),
                "direction_accuracy": 0.75,
                "normalized_l2": 0.5,
                "predicted_vitals": {"O2Sat": 91},
            },
        ],
    )

    summary = compare_three_eval_results(
        rule_csv,
        hybrid_csv,
        pure_llm_csv,
        output_dir,
    )

    assert (output_dir / PAIR_RESULTS_FILENAME).exists()
    assert (output_dir / SUMMARY_FILENAME).exists()
    assert json.loads((output_dir / SUMMARY_FILENAME).read_text()) == summary

    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert [row["pair_id"] for row in rows] == ["p1", "p2", "p3"]
    assert float(rows[0]["hybrid_delta_normalized_l2_vs_rule"]) == pytest.approx(-1.0)
    assert float(rows[0]["pure_llm_delta_normalized_l2_vs_rule"]) == pytest.approx(
        -1.75
    )
    assert float(rows[0]["pure_llm_delta_normalized_l2_vs_hybrid"]) == pytest.approx(
        -0.75
    )
    assert float(rows[0]["pure_llm_delta_direction_accuracy_vs_rule"]) == pytest.approx(
        0.5
    )
    assert rows[0]["best_engine_by_l2"] == "pure_llm"
    assert rows[0]["rule_passed"] == "False"
    assert rows[0]["hybrid_passed"] == "True"
    assert rows[0]["pure_llm_passed"] == "True"
    assert json.loads(rows[0]["failed_engines"]) == ["rule"]
    assert json.loads(rows[0]["passed_engines"]) == ["hybrid", "pure_llm"]
    assert rows[0]["failure_pattern"] == "rule_fail_hybrid_pass_pure_llm_pass"
    assert rows[1]["best_engine_by_l2"] == "rule"
    assert rows[2]["best_engine_by_l2"] == "tie"
    assert rows[2]["failure_pattern"] == "rule_fail_hybrid_fail_pure_llm_fail"
    assert json.loads(rows[0]["rule_predicted_vitals"]) == {"O2Sat": 84}
    assert json.loads(rows[0]["hybrid_predicted_vitals"]) == {"O2Sat": 90}
    assert json.loads(rows[0]["pure_llm_predicted_vitals"]) == {"O2Sat": 93}
    assert json.loads(rows[0]["hybrid_llm_adjustments"]) == {"O2Sat": 2}
    assert json.loads(rows[0]["hybrid_llm_reasoning"]) == {
        "O2Sat": "Hybrid corrected the rule oxygen response."
    }
    assert json.loads(rows[0]["pure_llm_reasoning"]) == {
        "O2Sat": "Pure LLM predicted improvement."
    }
    assert rows[2]["hybrid_llm_api_error"] == "hybrid error"
    assert rows[1]["pure_llm_llm_api_error"] == "pure error"
    assert "raw_l2" not in rows[0]

    overall = summary["overall"]
    assert overall["num_pairs"] == 3
    assert overall["engines"]["rule"]["normalized_l2_mean"] == pytest.approx(
        (2.0 + 1.0 + 0.5) / 3
    )
    assert overall["engines"]["hybrid"]["normalized_l2_median"] == pytest.approx(1.0)
    assert overall["engines"]["pure_llm"]["direction_accuracy_mean"] == pytest.approx(
        (1.0 + 0.0 + 0.75) / 3
    )
    assert overall["best_l2_counts"] == {
        "rule": 1,
        "hybrid": 0,
        "pure_llm": 1,
        "tie": 1,
        "none": 0,
    }
    assert overall["failure_analysis"]["success_criterion"] == (
        "direction_accuracy == 1.0 for every requested vital"
    )
    assert overall["failure_analysis"]["engine_failure_counts"] == {
        "rule": 2,
        "hybrid": 1,
        "pure_llm": 2,
    }
    assert overall["failure_analysis"]["rule_failed_hybrid_passed"]["count"] == 1
    assert overall["failure_analysis"]["all_three_failed"]["count"] == 1
    assert overall["failure_analysis"]["all_three_failed"]["pairs"][0]["pair_id"] == "p3"

    hybrid_vs_rule = overall["pairwise"]["hybrid_vs_rule"]
    assert hybrid_vs_rule["num_l2_compared_pairs"] == 3
    assert hybrid_vs_rule["improved_count"] == 1
    assert hybrid_vs_rule["worsened_count"] == 1
    assert hybrid_vs_rule["unchanged_count"] == 1
    assert hybrid_vs_rule["mean_delta_normalized_l2"] == pytest.approx(
        (-1.0 + 0.1 + 0.0) / 3
    )

    pure_vs_rule = overall["pairwise"]["pure_llm_vs_rule"]
    assert pure_vs_rule["improved_count"] == 1
    assert pure_vs_rule["worsened_count"] == 1
    assert pure_vs_rule["unchanged_count"] == 1
    assert pure_vs_rule["mean_delta_direction_accuracy"] == pytest.approx(
        (0.5 + -1.0 + 0.0) / 3
    )

    oxygen = summary["by_kind_hint"]["oxygen_support"]
    assert oxygen["count"] == 2
    assert oxygen["engines"]["pure_llm"]["normalized_l2_mean"] == pytest.approx(
        (0.25 + 0.5) / 2
    )

    console = capsys.readouterr().out
    assert "Compared pairs: 3" in console
    assert "Top 20 most improved pairs: hybrid vs rule" in console
    assert "Top 20 most improved pairs: pure llm vs rule" in console
    assert "transitions/case.json:p1" in console
    assert "Best engine by normalized_l2:" in console


def test_compare_three_uses_inner_join_on_source_case_and_pair(tmp_path: Path) -> None:
    rule_csv = tmp_path / "rule.csv"
    hybrid_csv = tmp_path / "hybrid.csv"
    pure_llm_csv = tmp_path / "pure.csv"
    output_dir = tmp_path / "out"
    _write_rows(rule_csv, [_base_row("matched"), _base_row("rule-only")])
    _write_rows(hybrid_csv, [_base_row("matched"), _base_row("hybrid-only")])
    _write_rows(pure_llm_csv, [_base_row("matched"), _base_row("pure-only")])

    summary = compare_three_eval_results(
        rule_csv,
        hybrid_csv,
        pure_llm_csv,
        output_dir,
        emit_console_summary=False,
    )

    assert summary["overall"]["num_pairs"] == 1
    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert [row["pair_id"] for row in rows] == ["matched"]


def test_duplicate_join_keys_raise_value_error(tmp_path: Path) -> None:
    rule_csv = tmp_path / "rule.csv"
    hybrid_csv = tmp_path / "hybrid.csv"
    pure_llm_csv = tmp_path / "pure.csv"
    output_dir = tmp_path / "out"
    _write_rows(rule_csv, [_base_row("p1")])
    _write_rows(hybrid_csv, [_base_row("p1")])
    _write_rows(pure_llm_csv, [_base_row("p1"), _base_row("p1")])

    with pytest.raises(ValueError, match="Duplicate pure_llm row"):
        compare_three_eval_results(
            rule_csv,
            hybrid_csv,
            pure_llm_csv,
            output_dir,
            emit_console_summary=False,
        )
