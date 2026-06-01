from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from ed_world_model.adapters.hybrid_physiology_adapter import (
    HybridPhysiologyAdapter,
)
from ed_world_model.state.global_state import GlobalState
from ed_world_model.state.state_manager import StateManager


class RecordingEngine:
    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.output = output or {
            "prediction": {
                "vitals": {
                    "HR": 100,
                    "BP_sys": 120,
                    "BP_dia": 70,
                    "RR": 20,
                    "O2Sat": 94,
                    "T": 37.0,
                },
                "features": {"oxygen_device": "NRB"},
            },
            "metadata": {"engine": "fake_hybrid"},
        }

    def predict(self, engine_input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(deepcopy(engine_input))
        return deepcopy(self.output)


class MutatingEngine:
    def predict(self, engine_input: dict[str, Any]) -> dict[str, Any]:
        engine_input["before"]["vitals"]["HR"] = 999
        engine_input["action"]["params"]["elapsed_min"] = 999
        return {
            "prediction": {
                "vitals": {"HR": 999},
                "features": {"mutated_engine_input": True},
            },
            "metadata": {"engine": "mutating_fake"},
        }


class EmptyAdjustmentLLMClient:
    model = "fake-hybrid-test-model"
    last_api_error = None

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"adjustments": {}}'


def _state() -> GlobalState:
    return GlobalState.model_validate(
        {
            "truth_state": {
                "scenario_description": "Safe runtime scenario context.",
                "demographics": {
                    "name": "Pat Doe",
                    "age": 64,
                    "sex": "F",
                    "weight_kg": 70,
                },
                "patient_internal_state": {
                    "chief_complaint": "shortness of breath",
                    "symptoms": ["dyspnea"],
                    "hidden_history": ["COPD"],
                    "hidden_allergies": ["latex"],
                    "hidden_home_medications": ["albuterol"],
                    "disclosure_rules": "Answer direct questions.",
                },
                "test_bank": [
                    {
                        "name": "Troponin",
                        "result": "SECRET_UNRELEASED_TROPONIN",
                    }
                ],
            },
            "patient_state": {
                "vitals": {
                    "HR": 118,
                    "BP_sys": 102,
                    "BP_dia": 62,
                    "RR": 30,
                    "O2Sat": 88,
                    "T": 37.3,
                },
                "features": {"rhythm": "sinus", "work_of_breathing": "high"},
                "status_flags": {
                    "is_alive": True,
                    "can_speak": True,
                    "is_conscious": True,
                },
            },
            "known_facts": {
                "chief_complaint": "shortness of breath",
                "known_history": [
                    {
                        "item": "asthma",
                        "status": "present",
                        "source_texts": ["I have asthma."],
                    }
                ],
                "known_allergies": [
                    {
                        "substance": "penicillin",
                        "status": "present",
                        "source_texts": ["I am allergic to penicillin."],
                    }
                ],
                "known_medications": [
                    {
                        "name": "inhaler",
                        "status": "current",
                        "source_texts": ["I use an inhaler."],
                    }
                ],
                "known_symptoms": [
                    {
                        "name": "wheezing",
                        "status": "present",
                        "source_texts": ["I am wheezing."],
                    }
                ],
                "available_results": [
                    {"name": "ECG", "result": "RELEASED_ECG_RESULT"}
                ],
            },
        }
    )


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(nested, key) for nested in value.values())
    if isinstance(value, list):
        return any(_contains_key(nested, key) for nested in value)
    return False


def _contains_value(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(nested, expected) for nested in value.values())
    if isinstance(value, list):
        return any(_contains_value(nested, expected) for nested in value)
    return False


def test_adapter_can_call_existing_hybrid_engine_with_fake_llm() -> None:
    llm_client = EmptyAdjustmentLLMClient()
    adapter = HybridPhysiologyAdapter(llm_client=llm_client)

    output = adapter.predict(
        _state(),
        {
            "raw_text": None,
            "kind_hint": "oxygen_support",
            "params": {"oxygen_device": "NRB"},
        },
    )

    assert output["metadata"]["engine"] == "hybrid"
    assert output["vitals"]["O2Sat"] == 94
    assert output["features"]["oxygen_device"] == "NRB"
    assert llm_client.prompts
    assert "raw_text: null" in llm_client.prompts[0]


def test_builds_engine_input_from_runtime_state_and_action() -> None:
    engine = RecordingEngine()
    adapter = HybridPhysiologyAdapter(engine)

    adapter.predict(
        _state(),
        {
            "raw_text": "do not forward",
            "kind_hint": "oxygen_support",
            "params": {"oxygen_device": "NRB", "FiO2": 1.0},
        },
    )

    engine_input = engine.calls[0]
    assert set(engine_input) == {
        "scenario_description",
        "case_context",
        "before",
        "action",
    }
    assert engine_input["scenario_description"] == "Safe runtime scenario context."
    assert engine_input["case_context"]["demographics"]["age"] == 64
    assert engine_input["case_context"]["patient_internal_state"][
        "hidden_history"
    ] == ["COPD"]
    assert engine_input["case_context"]["known_facts"]["known_history"] == [
        {
            "item": "asthma",
            "status": "present",
            "source_texts": ["I have asthma."],
        }
    ]
    assert engine_input["before"]["vitals"]["O2Sat"] == 88
    assert engine_input["before"]["features"]["work_of_breathing"] == "high"
    assert engine_input["before"]["status_flags"]["can_speak"] is True
    assert engine_input["action"] == {
        "raw_text": None,
        "kind_hint": "oxygen_support",
        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
    }


def test_released_results_only_enter_through_known_facts() -> None:
    engine = RecordingEngine()
    adapter = HybridPhysiologyAdapter(engine)

    adapter.predict(
        _state(),
        {"raw_text": None, "kind_hint": "no_action", "params": {"elapsed_min": 1}},
    )

    engine_input = engine.calls[0]
    assert not _contains_key(engine_input, "test_bank")
    assert not _contains_value(engine_input, "SECRET_UNRELEASED_TROPONIN")
    assert _contains_value(engine_input, "RELEASED_ECG_RESULT")


def test_adapter_does_not_add_transition_pair_fields_or_runtime_placeholders() -> None:
    engine = RecordingEngine()
    adapter = HybridPhysiologyAdapter(engine)

    adapter.predict(
        _state(),
        {"raw_text": None, "kind_hint": "no_action", "params": {"elapsed_min": 1}},
    )

    engine_input = engine.calls[0]
    for forbidden_key in (
        "label",
        "source",
        "modifier_text",
        "target",
        "evaluation",
        "case_id",
        "category",
        "pair_id",
    ):
        assert not _contains_key(engine_input, forbidden_key)


def test_supports_one_minute_no_action_with_null_raw_text() -> None:
    engine = RecordingEngine()
    adapter = HybridPhysiologyAdapter(engine)

    adapter.predict(
        _state(),
        {"raw_text": None, "kind_hint": "no_action", "params": {"elapsed_min": 1}},
    )

    assert engine.calls[0]["action"] == {
        "raw_text": None,
        "kind_hint": "no_action",
        "params": {"elapsed_min": 1},
    }


def test_adapter_does_not_mutate_global_state_or_action() -> None:
    state = _state()
    action = {
        "raw_text": None,
        "kind_hint": "no_action",
        "params": {"elapsed_min": 1},
    }
    state_before = state.model_dump()
    action_before = deepcopy(action)
    adapter = HybridPhysiologyAdapter(MutatingEngine())

    adapter.predict(state, action)

    assert state.model_dump() == state_before
    assert action == action_before


def test_existing_hybrid_path_handles_missing_null_vitals() -> None:
    llm_client = EmptyAdjustmentLLMClient()
    adapter = HybridPhysiologyAdapter(llm_client=llm_client)

    output = adapter.predict(
        GlobalState(),
        {"raw_text": None, "kind_hint": "no_action", "params": {"elapsed_min": 1}},
    )

    assert output["metadata"]["engine"] == "hybrid"
    assert output["vitals"] == {
        "HR": None,
        "BP_sys": None,
        "BP_dia": None,
        "RR": None,
        "O2Sat": None,
        "T": None,
    }


def test_normalized_output_can_be_applied_by_state_manager_fields() -> None:
    adapter = HybridPhysiologyAdapter(
        RecordingEngine(
            {
                "prediction": {
                    "vitals": {
                        "HR": 101,
                        "BP_sys": 121,
                        "BP_dia": 71,
                        "RR": 21,
                        "O2Sat": 95,
                        "T": 37.1,
                    },
                    "features": {"oxygen_device": "NRB"},
                },
                "metadata": {"engine": "fake_hybrid"},
            }
        )
    )
    manager = StateManager(_state())

    output = adapter.predict(
        manager.state,
        {
            "raw_text": None,
            "kind_hint": "oxygen_support",
            "params": {"oxygen_device": "NRB"},
        },
    )
    updated_patient_state = manager.apply_physiology_update(
        vitals=output["vitals"],
        features=output["features"],
    )

    assert set(output) == {"vitals", "features", "metadata"}
    assert updated_patient_state.vitals.O2Sat == 95
    assert updated_patient_state.features.oxygen_device == "NRB"


def test_adapter_rejects_missing_engine_dependency() -> None:
    with pytest.raises(ValueError, match="requires an engine or llm_client"):
        HybridPhysiologyAdapter()
