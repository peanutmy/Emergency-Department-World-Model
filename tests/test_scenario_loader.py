from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.scenario_loader as scenario_loader_module
from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.scenario_loader import (
    DEFAULT_DISCLOSURE_RULES,
    ScenarioLoader,
    load_scenario,
)
from ed_world_model.state.global_state import GlobalState


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "scenario_loader_minimal.json"


def _scenario() -> dict[str, Any]:
    return {
        "schema_version": "educational_transition_pair_v14",
        "case_id": "loader_test_case",
        "category": "Testing",
        "source_pdf": "ignored.pdf",
        "chief_complaint": "Top-level complaint should be ignored",
        "Symptoms": ["Top-level symptom should be ignored"],
        "scenario_description": "A compact ScenarioLoader test scenario.",
        "case_context": {
            "demographics": {
                "name": "Pat Example",
                "age": 42,
                "sex": "F",
                "weight_kg": 68,
                "ignored_extra_demographic": "not modeled",
            },
            "history": ["asthma", "hypertension"],
            "allergies": ["penicillin"],
            "home_medications": ["albuterol"],
            "chief_complaint": "shortness of breath",
            "symptoms": ["dyspnea", "wheezing"],
            "disclosure_rules": "Answer direct questions using hidden truth.",
            "baseline_vitals": {
                "HR": 118,
                "BP_sys": 102,
                "BP_dia": 62,
                "RR": 30,
                "O2Sat": 88,
                "T": 37.3,
            },
            "baseline_features": {
                "rhythm": "sinus tachycardia",
                "work_of_breathing": "high",
                "oxygen_device": "nasal cannula",
            },
            "supporting_findings": [
                {
                    "test_code": "ECG_CODE_SHOULD_NOT_STORE_WHEN_NAME_EXISTS",
                    "test_name": "Initial ECG",
                    "test_category": "ecg",
                    "result_summary": "Sinus tachycardia.",
                    "turnaround_turns": 1,
                },
                {
                    "test_code": "CXR",
                    "test_category": "imaging",
                    "result_summary": "No focal consolidation.",
                },
                {
                    "test_category": "lab",
                    "result_summary": "Skipped because no name or code.",
                },
            ],
        },
        "pairs": [
            {
                "source": {
                    "modifier_text": "PAIR_MODIFIER_TEXT_SHOULD_NOT_LOAD",
                },
                "input": {
                    "before": {
                        "vitals": {
                            "HR": 999,
                            "BP_sys": 999,
                            "O2Sat": 12,
                        },
                        "features": {
                            "pair_only_feature": "PAIR_FEATURE_SHOULD_NOT_LOAD",
                        },
                    },
                    "action": {
                        "raw_text": "PAIR_RAW_TEXT_SHOULD_NOT_LOAD",
                        "kind_hint": "oxygen_support",
                        "params": {"oxygen_device": "NRB"},
                    },
                },
                "label": {
                    "target": {
                        "vitals": {
                            "HR": 1,
                        },
                        "features": {
                            "target_feature": "PAIR_TARGET_SHOULD_NOT_LOAD",
                        },
                    },
                    "evaluation": "PAIR_EVALUATION_SHOULD_NOT_LOAD",
                },
            }
        ],
    }


def _load(scenario: dict[str, Any] | None = None) -> GlobalState:
    return ScenarioLoader().load_dict(_scenario() if scenario is None else scenario)


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _contains_value(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


def test_load_minimal_scenario_dict_into_global_state() -> None:
    state = _load()

    assert isinstance(state, GlobalState)
    assert state.runtime_state.turn_index == 0


def test_scenario_description_is_populated() -> None:
    state = _load()

    assert (
        state.truth_state.scenario_description
        == "A compact ScenarioLoader test scenario."
    )


def test_demographics_are_populated_from_case_context() -> None:
    state = _load()

    assert state.truth_state.demographics.name == "Pat Example"
    assert state.truth_state.demographics.age == 42
    assert state.truth_state.demographics.sex == "F"
    assert state.truth_state.demographics.weight_kg == 68


def test_patient_internal_hidden_fields_populate_from_case_context() -> None:
    state = _load()
    internal = state.truth_state.patient_internal_state

    assert internal.hidden_history == ["asthma", "hypertension"]
    assert internal.hidden_allergies == ["penicillin"]
    assert internal.hidden_home_medications == ["albuterol"]


def test_patient_internal_chief_complaint_uses_case_context_value() -> None:
    state = _load()

    assert state.truth_state.patient_internal_state.chief_complaint == (
        "shortness of breath"
    )


def test_patient_internal_symptoms_use_case_context_value() -> None:
    state = _load()

    assert state.truth_state.patient_internal_state.symptoms == [
        "dyspnea",
        "wheezing",
    ]


def test_missing_chief_complaint_defaults_to_none() -> None:
    scenario = _scenario()
    del scenario["case_context"]["chief_complaint"]

    state = _load(scenario)

    assert state.truth_state.patient_internal_state.chief_complaint is None


def test_missing_symptoms_defaults_to_empty_list() -> None:
    scenario = _scenario()
    del scenario["case_context"]["symptoms"]

    state = _load(scenario)

    assert state.truth_state.patient_internal_state.symptoms == []


def test_missing_disclosure_rules_uses_simple_default_prompt_guidance() -> None:
    scenario = _scenario()
    del scenario["case_context"]["disclosure_rules"]

    state = _load(scenario)

    assert state.truth_state.patient_internal_state.disclosure_rules == (
        DEFAULT_DISCLOSURE_RULES
    )


def test_supporting_findings_become_test_bank_items() -> None:
    state = _load()

    assert [item.model_dump() for item in state.truth_state.test_bank] == [
        {
            "name": "Initial ECG",
            "result": "Sinus tachycardia.",
            "turnaround_turns": 1,
        },
        {
            "name": "CXR",
            "result": "No focal consolidation.",
            "turnaround_turns": DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS,
        },
    ]


def test_missing_turnaround_turns_uses_default_turnaround() -> None:
    state = _load()

    assert (
        state.truth_state.test_bank[1].turnaround_turns
        == DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
    )


def test_baseline_vitals_are_initialized_from_case_context_baseline_vitals() -> None:
    state = _load()

    assert state.patient_state.vitals.model_dump() == {
        "HR": 118.0,
        "BP_sys": 102.0,
        "BP_dia": 62.0,
        "RR": 30.0,
        "O2Sat": 88.0,
        "T": 37.3,
    }


def test_baseline_features_are_initialized_from_case_context_baseline_features() -> None:
    state = _load()

    assert state.patient_state.features.model_dump() == {
        "rhythm": "sinus tachycardia",
        "work_of_breathing": "high",
        "oxygen_device": "nasal cannula",
    }


def test_missing_baseline_vitals_produces_null_vitals_without_crashing() -> None:
    scenario = _scenario()
    del scenario["case_context"]["baseline_vitals"]

    state = _load(scenario)

    assert state.patient_state.vitals.model_dump(exclude_none=True) == {}


def test_missing_baseline_features_produces_empty_features_without_crashing() -> None:
    scenario = _scenario()
    del scenario["case_context"]["baseline_features"]

    state = _load(scenario)

    assert state.patient_state.features.model_dump() == {}


def test_pairs_are_completely_ignored() -> None:
    state = _load()
    dumped = state.model_dump()

    assert "pairs" not in dumped
    assert not _contains_value(dumped, "PAIR_FEATURE_SHOULD_NOT_LOAD")
    assert not _contains_value(dumped, "PAIR_RAW_TEXT_SHOULD_NOT_LOAD")


def test_pairs_input_before_is_not_used_even_when_baseline_vitals_missing() -> None:
    scenario = _scenario()
    del scenario["case_context"]["baseline_vitals"]

    state = _load(scenario)

    assert state.patient_state.vitals.HR is None
    assert not _contains_value(state.model_dump(), 999)
    assert not _contains_value(state.model_dump(), 12)


def test_pair_labels_source_modifier_text_and_target_are_not_copied() -> None:
    state = _load()
    dumped = state.model_dump()

    assert not _contains_value(dumped, "PAIR_MODIFIER_TEXT_SHOULD_NOT_LOAD")
    assert not _contains_value(dumped, "PAIR_EVALUATION_SHOULD_NOT_LOAD")
    assert not _contains_value(dumped, "PAIR_TARGET_SHOULD_NOT_LOAD")


def test_known_facts_available_results_starts_empty() -> None:
    state = _load()

    assert state.known_facts.available_results == []


def test_known_facts_clinical_fact_lists_start_empty() -> None:
    state = _load()

    assert state.known_facts.known_history == []
    assert state.known_facts.known_allergies == []
    assert state.known_facts.known_medications == []
    assert state.known_facts.known_symptoms == []


def test_runtime_pending_diagnostic_results_starts_empty() -> None:
    state = _load()

    assert state.runtime_state.pending_diagnostic_results == []


def test_runtime_raw_text_is_not_required_anywhere() -> None:
    state = _load()

    assert not _contains_key(state.model_dump(), "raw_text")


def test_missing_supporting_findings_produces_empty_test_bank() -> None:
    scenario = _scenario()
    del scenario["case_context"]["supporting_findings"]

    state = _load(scenario)

    assert state.truth_state.test_bank == []


def test_loading_from_dict_works() -> None:
    state = ScenarioLoader().load(_scenario(), max_turns=7)

    assert state.truth_state.demographics.name == "Pat Example"
    assert state.runtime_state.max_turns == 7


def test_loading_from_file_path_works() -> None:
    state = ScenarioLoader().load_file(FIXTURE_PATH)

    assert state.truth_state.scenario_description == (
        "A small fixture scenario for ScenarioLoader tests."
    )
    assert state.truth_state.demographics.name == "Fixture Patient"
    assert state.truth_state.test_bank[0].name == "Initial ECG"
    assert state.patient_state.vitals.HR == 104
    assert state.patient_state.features.model_dump()["rhythm"] == (
        "sinus tachycardia"
    )


def test_loader_does_not_mutate_input_dict() -> None:
    scenario = _scenario()
    original = deepcopy(scenario)

    ScenarioLoader().load_dict(scenario)

    assert scenario == original


def test_loader_does_not_import_transition_engines() -> None:
    source = inspect.getsource(scenario_loader_module)

    assert "transition_engines" not in source
    assert "transition_engines" not in sys.modules


def test_convenience_load_scenario_dispatches_to_loader() -> None:
    state = load_scenario(_scenario())

    assert isinstance(state, GlobalState)
    assert state.truth_state.demographics.age == 42
