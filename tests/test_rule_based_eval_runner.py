from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transition_engines.run_rule_based_eval as runner
from transition_engines.run_rule_based_eval import (
    PAIR_RESULTS_FILENAME,
    SUMMARY_FILENAME,
    run_rule_based_eval,
)


BASE_VITALS = {
    "HR": 100,
    "BP_sys": 120,
    "BP_dia": 70,
    "RR": 20,
    "O2Sat": 94,
    "T": 37.0,
}


def _case_doc(case_id: str, pairs: list[dict]) -> dict:
    return {
        "schema_version": "synthetic_v1",
        "case_id": case_id,
        "category": "synthetic",
        "source_pdf": f"{case_id}.pdf",
        "scenario_description": "Synthetic case.",
        "case_context": {
            "demographics": {"age": 50},
            "history": [],
            "home_medications": [],
            "allergies": [],
            "baseline_features": {},
            "supporting_findings": [{"result_summary": "answer leak"}],
        },
        "pairs": pairs,
    }


def _pair(
    pair_id: str,
    kind_hint: str,
    *,
    before_vitals: dict | None = None,
    params: dict | None = None,
    target_vitals: list[str],
    target_values: dict,
    pair_type: str = "physiology_response",
) -> dict:
    return {
        "id": pair_id,
        "pair_type": pair_type,
        "source": {"modifier_text": "target value leak"},
        "input": {
            "before": {
                "vitals": dict(before_vitals or BASE_VITALS),
                "features": {"rhythm": "sinus"},
            },
            "action": {
                "raw_text": kind_hint,
                "kind_hint": kind_hint,
                "params": params or {},
            },
        },
        "label": {
            "target": {
                "vitals": dict(target_values),
                "features": {"rhythm": "answer feature"},
            },
            "evaluation": {
                "target_vitals": list(target_vitals),
                "target_features": ["rhythm"],
            },
        },
    }


def _write_case(path: Path, case_doc: dict) -> None:
    path.write_text(json.dumps(case_doc), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(nested, key) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(nested, key) for nested in value)
    return False


def test_single_file_input_writes_csv_and_summary_json(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "single-case",
            [
                _pair(
                    "p1",
                    "oxygen_support",
                    before_vitals={**BASE_VITALS, "O2Sat": 88, "RR": 28},
                    params={"oxygen_device": "NRB"},
                    target_vitals=["O2Sat"],
                    target_values={"O2Sat": 94},
                )
            ],
        ),
    )

    summary = run_rule_based_eval(case_path, output_dir, emit_console_summary=False)

    csv_path = output_dir / PAIR_RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    assert csv_path.exists()
    assert summary_path.exists()
    assert summary["overall"]["num_pairs"] == 1
    assert summary["by_kind_hint"]["oxygen_support"]["count"] == 1

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary == summary

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "single-case"
    assert rows[0]["pair_id"] == "p1"
    assert json.loads(rows[0]["target_vitals"]) == ["O2Sat"]
    assert json.loads(rows[0]["predicted_vitals"])["O2Sat"] == 94


def test_directory_input_writes_outputs_and_aggregates_kind_hint(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "cases"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_case(
        input_dir / "a.json",
        _case_doc(
            "case-a",
            [
                _pair(
                    "p1",
                    "oxygen_support",
                    before_vitals={**BASE_VITALS, "O2Sat": 88},
                    params={"oxygen_device": "NRB"},
                    target_vitals=["O2Sat"],
                    target_values={"O2Sat": 94},
                ),
                _pair(
                    "p2",
                    "no_action",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                    pair_type="time_progression",
                    params={"elapsed_min": 0},
                ),
            ],
        ),
    )
    _write_case(
        input_dir / "b.json",
        _case_doc(
            "case-b",
            [
                _pair(
                    "p1",
                    "oxygen_support",
                    before_vitals={**BASE_VITALS, "O2Sat": 92},
                    params={"oxygen_device": "NRB"},
                    target_vitals=["O2Sat"],
                    target_values={"O2Sat": 94},
                )
            ],
        ),
    )
    (input_dir / "schema.json").write_text('{"title": "not a case"}', encoding="utf-8")

    summary = run_rule_based_eval(input_dir, output_dir, emit_console_summary=False)

    assert (output_dir / PAIR_RESULTS_FILENAME).exists()
    assert (output_dir / SUMMARY_FILENAME).exists()
    assert summary["overall"]["num_pairs"] == 3
    assert summary["by_kind_hint"]["oxygen_support"]["count"] == 2
    assert summary["by_kind_hint"]["no_action"]["count"] == 1
    assert summary["by_pair_type"]["physiology_response"]["count"] == 2
    assert summary["by_pair_type"]["time_progression"]["count"] == 1


def test_recursive_directory_input_finds_nested_case_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "cases"
    nested_dir = input_dir / "nested"
    output_dir = tmp_path / "out"
    nested_dir.mkdir(parents=True)
    _write_case(
        nested_dir / "nested.json",
        _case_doc(
            "nested-case",
            [
                _pair(
                    "p1",
                    "no_action",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                    params={"elapsed_min": 0},
                    pair_type="time_progression",
                )
            ],
        ),
    )

    summary = run_rule_based_eval(
        input_dir,
        output_dir,
        recursive=True,
        emit_console_summary=False,
    )

    assert summary["overall"]["num_pairs"] == 1
    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert rows[0]["case_id"] == "nested-case"


def test_target_vital_only_evaluation_is_used(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "target-only",
            [
                _pair(
                    "p1",
                    "not_a_rule",
                    target_vitals=["HR"],
                    target_values={"HR": 100, "BP_sys": 999},
                )
            ],
        ),
    )

    summary = run_rule_based_eval(case_path, output_dir, emit_console_summary=False)

    assert summary["overall"]["normalized_l2_mean"] == pytest.approx(0.0)
    assert summary["by_target_vital"]["HR"]["count"] == 1
    assert summary["by_target_vital"]["HR"][
        "normalized_abs_error_mean_for_that_vital"
    ] == pytest.approx(0.0)
    assert summary["by_target_vital"]["BP_sys"]["count"] == 0

    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert json.loads(rows[0]["target_vitals"]) == ["HR"]
    assert json.loads(rows[0]["target_vitals_values"]) == {"HR": 100}
    assert float(rows[0]["normalized_l2"]) == pytest.approx(0.0)


def test_runner_passes_only_sanitized_builder_input_to_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "sanitize-case",
            [
                _pair(
                    "p1",
                    "not_a_rule",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                )
            ],
        ),
    )
    observed_inputs: list[dict] = []

    class SpyEngine:
        def predict(self, engine_input: dict) -> dict:
            observed_inputs.append(engine_input)
            return {"prediction": {"vitals": dict(engine_input["before"]["vitals"])}}

    monkeypatch.setattr(runner, "RuleBasedEngine", lambda: SpyEngine())

    run_rule_based_eval(case_path, output_dir, emit_console_summary=False)

    assert len(observed_inputs) == 1
    engine_input = observed_inputs[0]
    assert "label" not in engine_input
    assert "source" not in engine_input
    assert "supporting_findings" not in engine_input["case_context"]
    assert not _contains_key(engine_input, "modifier_text")
