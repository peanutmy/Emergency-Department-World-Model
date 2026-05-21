from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.compare_eval_results import (
    PAIR_RESULTS_FILENAME,
    SUMMARY_FILENAME,
    compare_eval_results,
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
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("target_vitals", "predicted_vitals", "llm_adjustments"):
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
    }


def test_compare_eval_results_writes_pair_csv_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rule_csv = tmp_path / "rule_based_pair_results.csv"
    hybrid_csv = tmp_path / "hybrid_pair_results.csv"
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
                "direction_accuracy": 0.75,
                "normalized_l2": 1.0,
                "predicted_vitals": {"O2Sat": 90},
                "llm_adjustments": {"O2Sat": 2},
            },
            {
                **_base_row("p2", kind_hint="no_action"),
                "direction_accuracy": 1.0,
                "normalized_l2": 1.0000000005,
                "predicted_vitals": {"O2Sat": 89},
                "llm_adjustments": {"O2Sat": None},
            },
            {
                **_base_row("p3"),
                "direction_accuracy": 0.25,
                "normalized_l2": 0.75,
                "predicted_vitals": {"O2Sat": 88},
                "llm_adjustments": {"O2Sat": -3},
            },
        ],
    )

    summary = compare_eval_results(rule_csv, hybrid_csv, output_dir)

    assert (output_dir / PAIR_RESULTS_FILENAME).exists()
    assert (output_dir / SUMMARY_FILENAME).exists()
    assert json.loads((output_dir / SUMMARY_FILENAME).read_text()) == summary

    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert [row["pair_id"] for row in rows] == ["p1", "p2", "p3"]
    assert float(rows[0]["delta_normalized_l2"]) == pytest.approx(-1.0)
    assert float(rows[0]["delta_direction_accuracy"]) == pytest.approx(0.25)
    assert json.loads(rows[0]["rule_predicted_vitals"]) == {"O2Sat": 84}
    assert json.loads(rows[0]["hybrid_predicted_vitals"]) == {"O2Sat": 90}
    assert json.loads(rows[0]["hybrid_llm_adjustments"]) == {"O2Sat": 2}
    assert "raw_l2" not in rows[0]

    overall = summary["overall"]
    assert overall["num_pairs"] == 3
    assert overall["num_l2_compared_pairs"] == 3
    assert overall["mean_delta_normalized_l2"] == pytest.approx(
        (-1.0 + 0.0000000005 + 0.25) / 3
    )
    assert overall["median_delta_normalized_l2"] == pytest.approx(0.0000000005)
    assert overall["improved_count"] == 1
    assert overall["worsened_count"] == 1
    assert overall["unchanged_count"] == 1
    assert overall["improved_percentage"] == pytest.approx(100 / 3)
    assert overall["worsened_percentage"] == pytest.approx(100 / 3)
    assert overall["unchanged_percentage"] == pytest.approx(100 / 3)

    oxygen = summary["by_kind_hint"]["oxygen_support"]
    assert oxygen["count"] == 2
    assert oxygen["improved_count"] == 1
    assert oxygen["worsened_count"] == 1
    assert oxygen["unchanged_count"] == 0
    assert oxygen["mean_delta_l2"] == pytest.approx((-1.0 + 0.25) / 2)
    assert oxygen["median_delta_l2"] == pytest.approx((-1.0 + 0.25) / 2)
    assert oxygen["mean_delta_direction_accuracy"] == pytest.approx(
        (0.25 + -0.5) / 2
    )

    console = capsys.readouterr().out
    assert "Top 20 most improved pairs:" in console
    assert "transitions/case.json:p1" in console
    assert "Top 20 most worsened pairs:" in console
    assert "transitions/case.json:p3" in console
    assert "Top 20 high-error unchanged pairs:" in console
    assert "transitions/case.json:p2" in console


def test_compare_uses_inner_join_on_source_case_and_pair(tmp_path: Path) -> None:
    rule_csv = tmp_path / "rule.csv"
    hybrid_csv = tmp_path / "hybrid.csv"
    output_dir = tmp_path / "out"
    _write_rows(
        rule_csv,
        [
            _base_row("matched"),
            _base_row("rule-only"),
        ],
    )
    _write_rows(
        hybrid_csv,
        [
            _base_row("matched"),
            _base_row("hybrid-only"),
        ],
    )

    summary = compare_eval_results(
        rule_csv,
        hybrid_csv,
        output_dir,
        emit_console_summary=False,
    )

    assert summary["overall"]["num_pairs"] == 1
    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert [row["pair_id"] for row in rows] == ["matched"]


def test_duplicate_join_keys_raise_value_error(tmp_path: Path) -> None:
    rule_csv = tmp_path / "rule.csv"
    hybrid_csv = tmp_path / "hybrid.csv"
    output_dir = tmp_path / "out"
    _write_rows(rule_csv, [_base_row("p1"), _base_row("p1")])
    _write_rows(hybrid_csv, [_base_row("p1")])

    with pytest.raises(ValueError, match="Duplicate rule row"):
        compare_eval_results(
            rule_csv,
            hybrid_csv,
            output_dir,
            emit_console_summary=False,
        )
