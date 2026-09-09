from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transition_engines.run_pure_llm_eval as runner
from transition_engines.run_pure_llm_eval import (
    CSV_COLUMNS,
    PAIR_RESULTS_FILENAME,
    SUMMARY_FILENAME,
    FakeLLMClient,
    run_pure_llm_eval,
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
            "supporting_findings": [
                {"result_summary": "SUPPORTING_FINDINGS_LEAK"}
            ],
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
        "source": {"modifier_text": "SOURCE_MODIFIER_TEXT_LEAK"},
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
                "features": {"rhythm": "LABEL_TARGET_FEATURE_LEAK"},
            },
            "leak": "LABEL_LEAK_VALUE",
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


def test_single_file_input_writes_csv_and_summary_json(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    before_vitals = {**BASE_VITALS, "O2Sat": 88, "RR": 28}
    _write_case(
        case_path,
        _case_doc(
            "single-case",
            [
                _pair(
                    "p1",
                    "oxygen_support",
                    before_vitals=before_vitals,
                    params={"oxygen_device": "NRB"},
                    target_vitals=["O2Sat"],
                    target_values={"O2Sat": 94},
                )
            ],
        ),
    )

    summary = run_pure_llm_eval(case_path, output_dir, emit_console_summary=False)

    csv_path = output_dir / PAIR_RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    assert csv_path.exists()
    assert summary_path.exists()
    assert summary["overall"]["num_pairs"] == 1
    assert summary["overall"]["num_api_errors"] == 0
    assert summary["overall"]["num_parse_errors"] == 0
    assert summary["by_kind_hint"]["oxygen_support"]["count"] == 1
    assert summary["by_kind_hint"]["oxygen_support"]["num_api_errors"] == 0
    assert summary["by_kind_hint"]["oxygen_support"]["num_parse_errors"] == 0

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary == summary

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "single-case"
    assert rows[0]["pair_id"] == "p1"
    assert json.loads(rows[0]["target_vitals"]) == ["O2Sat"]
    assert json.loads(rows[0]["before_vitals"]) == before_vitals
    assert json.loads(rows[0]["predicted_vitals"]) == before_vitals
    assert json.loads(rows[0]["target_vitals_values"]) == {"O2Sat": 94}
    assert json.loads(rows[0]["llm_raw_response"]) == {"vitals": {}}
    assert rows[0]["llm_parse_error"] == ""


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

    summary = run_pure_llm_eval(input_dir, output_dir, emit_console_summary=False)

    assert (output_dir / PAIR_RESULTS_FILENAME).exists()
    assert (output_dir / SUMMARY_FILENAME).exists()
    assert summary["overall"]["num_pairs"] == 3
    assert summary["by_kind_hint"]["oxygen_support"]["count"] == 2
    assert summary["by_kind_hint"]["no_action"]["count"] == 1
    assert summary["by_pair_type"]["physiology_response"]["count"] == 2
    assert summary["by_pair_type"]["time_progression"]["count"] == 1


def test_fake_mode_instantiates_without_api_key_and_copies_before_vitals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    before_vitals = {**BASE_VITALS, "BP_sys": 80, "BP_dia": 50, "HR": 125}
    _write_case(
        case_path,
        _case_doc(
            "fake-no-key-case",
            [
                _pair(
                    "p1",
                    "fluid_bolus",
                    before_vitals=before_vitals,
                    params={"volume_ml": 1000},
                    target_vitals=["HR", "BP_sys", "BP_dia"],
                    target_values={"HR": 120, "BP_sys": 95, "BP_dia": 58},
                )
            ],
        ),
    )

    summary = run_pure_llm_eval(
        case_path,
        output_dir,
        llm_mode="fake",
        emit_console_summary=False,
    )

    assert summary["overall"]["num_pairs"] == 1
    row = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)[0]
    assert json.loads(row["predicted_vitals"]) == before_vitals
    assert row["model"] == "fake"


def test_target_vital_names_are_passed_to_pure_llm_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "target-pass-case",
            [
                _pair(
                    "p1",
                    "no_action",
                    target_vitals=["HR", "O2Sat"],
                    target_values={"HR": 100, "O2Sat": 94},
                    params={"elapsed_min": 0},
                )
            ],
        ),
    )
    observed_target_vitals: list[list[str]] = []

    class SpyPureLLMEngine:
        def __init__(self, llm_client: object) -> None:
            pass

        def predict(
            self,
            engine_input: dict,
            target_vital_names: list[str] | None = None,
        ) -> dict:
            observed_target_vitals.append(list(target_vital_names or []))
            return {
                "prediction": {"vitals": dict(engine_input["before"]["vitals"])},
                "metadata": {},
            }

    monkeypatch.setattr(runner, "PureLLMEngine", SpyPureLLMEngine)

    run_pure_llm_eval(case_path, output_dir, emit_console_summary=False)

    assert observed_target_vitals == [["HR", "O2Sat"]]


def test_runner_prompt_has_no_forbidden_leakage(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    llm = FakeLLMClient()
    _write_case(
        case_path,
        _case_doc(
            "leakage-case",
            [
                _pair(
                    "p1",
                    "oxygen_support",
                    before_vitals={**BASE_VITALS, "O2Sat": 88},
                    params={
                        "oxygen_device": "NRB",
                        "modifier_text": "PARAM_MODIFIER_TEXT_LEAK",
                        "supporting_findings": "PARAM_FINDINGS_LEAK",
                        "rule_prediction": "RULE_PREDICTION_LEAK",
                    },
                    target_vitals=["O2Sat"],
                    target_values={"O2Sat": 99.1234},
                )
            ],
        ),
    )

    run_pure_llm_eval(
        case_path,
        output_dir,
        llm_client=llm,
        emit_console_summary=False,
    )

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "LABEL_LEAK_VALUE" not in prompt
    assert "LABEL_TARGET_FEATURE_LEAK" not in prompt
    assert "SOURCE_MODIFIER_TEXT_LEAK" not in prompt
    assert "SUPPORTING_FINDINGS_LEAK" not in prompt
    assert "PARAM_MODIFIER_TEXT_LEAK" not in prompt
    assert "PARAM_FINDINGS_LEAK" not in prompt
    assert "RULE_PREDICTION_LEAK" not in prompt
    assert "99.1234" not in prompt
    for forbidden_key in (
        "label",
        "target",
        "source",
        "modifier_text",
        "supporting_findings",
        "rule_prediction",
    ):
        assert forbidden_key not in prompt


def test_make_real_llm_client_uses_vitals_schema_and_cli_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class SpyOpenAILLMClient:
        def __init__(
            self,
            *,
            model: str,
            temperature: float,
            max_output_tokens: int,
            response_schema_type: str,
        ) -> None:
            observed["model"] = model
            observed["temperature"] = temperature
            observed["max_output_tokens"] = max_output_tokens
            observed["response_schema_type"] = response_schema_type

    monkeypatch.setattr(runner, "OpenAILLMClient", SpyOpenAILLMClient)

    client = runner._make_llm_client(
        "real",
        model="gpt-test",
        temperature=0.4,
        max_output_tokens=77,
    )

    assert isinstance(client, SpyOpenAILLMClient)
    assert observed == {
        "model": "gpt-test",
        "temperature": 0.4,
        "max_output_tokens": 77,
        "response_schema_type": "vitals",
    }


def test_llm_api_error_is_recorded_in_pair_output_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "api-error-case",
            [
                _pair(
                    "p1",
                    "no_action",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                    params={"elapsed_min": 0},
                )
            ],
        ),
    )

    class SpyPureLLMEngine:
        def __init__(self, llm_client: object) -> None:
            pass

        def predict(
            self,
            engine_input: dict,
            target_vital_names: list[str] | None = None,
        ) -> dict:
            return {
                "prediction": {"vitals": dict(engine_input["before"]["vitals"])},
                "metadata": {
                    "llm_raw_response": '{"vitals": {}}',
                    "llm_parse_error": None,
                    "llm_api_error": "RuntimeError: redacted",
                    "model": "gpt-test",
                },
            }

    monkeypatch.setattr(runner, "PureLLMEngine", SpyPureLLMEngine)

    summary = run_pure_llm_eval(case_path, output_dir, emit_console_summary=False)

    row = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)[0]
    assert row["llm_api_error"] == "RuntimeError: redacted"
    assert summary["overall"]["num_api_errors"] == 1
    assert summary["by_kind_hint"]["no_action"]["num_api_errors"] == 1


def test_no_raw_l2_or_feature_evaluation_outputs(tmp_path: Path) -> None:
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

    summary = run_pure_llm_eval(case_path, output_dir, emit_console_summary=False)

    assert "raw_l2" not in json.dumps(summary, sort_keys=True)
    assert "feature" not in json.dumps(summary, sort_keys=True).lower()
    assert "raw_l2" not in CSV_COLUMNS
    assert all("feature" not in column for column in CSV_COLUMNS)
    assert summary["by_target_vital"]["HR"]["count"] == 1
    assert summary["by_target_vital"]["BP_sys"]["count"] == 0

    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert "raw_l2" not in rows[0]
    assert all("feature" not in column for column in rows[0])
    assert json.loads(rows[0]["target_vitals"]) == ["HR"]
    assert json.loads(rows[0]["target_vitals_values"]) == {"HR": 100}


def test_limit_pairs_evaluates_only_prefix(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    output_dir = tmp_path / "out"
    _write_case(
        case_path,
        _case_doc(
            "limit-case",
            [
                _pair(
                    "p1",
                    "no_action",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                    params={"elapsed_min": 0},
                ),
                _pair(
                    "p2",
                    "no_action",
                    target_vitals=["HR"],
                    target_values={"HR": 100},
                    params={"elapsed_min": 0},
                ),
            ],
        ),
    )

    summary = run_pure_llm_eval(
        case_path,
        output_dir,
        limit_pairs=1,
        emit_console_summary=False,
    )

    assert summary["overall"]["num_pairs"] == 1
    rows = _read_csv_rows(output_dir / PAIR_RESULTS_FILENAME)
    assert [row["pair_id"] for row in rows] == ["p1"]
