from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.common import CANONICAL_VITAL_KEYS
from transition_engines.pure_llm_engine import (
    PureLLMEngine,
    build_llm_prompt,
    parse_llm_vitals,
)


BASE_VITALS = {
    "HR": 100,
    "BP_sys": 120,
    "BP_dia": 70,
    "RR": 20,
    "O2Sat": 84,
    "T": 37.0,
}


class FakeLLMClient:
    def __init__(self, response: str, model: str = "fake-model") -> None:
        self.response = response
        self.model = model
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RaisingLLMClient:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("request failed for secret-api-key")


def _engine_input(
    *,
    vitals: dict | None = None,
    kind_hint: str | None = "oxygen_support",
    params: dict | None = None,
) -> dict:
    return {
        "case_id": "case-1",
        "category": "Respiratory",
        "scenario_description": "Synthetic safe scenario.",
        "case_context": {
            "demographics": {"age": 44},
            "history": ["shortness of breath"],
        },
        "before": {
            "vitals": dict(vitals or BASE_VITALS),
            "features": {"rhythm": "sinus"},
        },
        "action": {
            "raw_text": "apply oxygen",
            "kind_hint": kind_hint,
            "params": params or {"oxygen_device": "NRB"},
        },
    }


def test_llm_vital_prediction_updates_requested_vital_and_copies_others() -> None:
    llm = FakeLLMClient(
        '{"vitals": {"O2Sat": 88}, '
        '"reasoning": {"O2Sat": "Oxygen should partially improve saturation."}}'
    )
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"]["O2Sat"] == 88
    for vital in set(CANONICAL_VITAL_KEYS) - {"O2Sat"}:
        assert output["prediction"]["vitals"][vital] == BASE_VITALS[vital]
    assert output["prediction"]["features"]["oxygen_device"] == "NRB"
    assert output["prediction"]["features"]["FiO2"] == 0.8
    assert output["metadata"]["llm_reasoning"]["O2Sat"] == (
        "Oxygen should partially improve saturation."
    )


def test_non_requested_llm_prediction_is_ignored() -> None:
    llm = FakeLLMClient('{"vitals": {"HR": 130, "O2Sat": 88}}')
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"]["HR"] == BASE_VITALS["HR"]
    assert output["prediction"]["vitals"]["O2Sat"] == 88


def test_feature_prediction_is_rule_based_not_llm_based() -> None:
    llm = FakeLLMClient(
        '{"vitals": {"O2Sat": 88}, "features": {"oxygen_device": "room_air"}}'
    )
    engine = PureLLMEngine(llm)

    output = engine.predict(
        _engine_input(
            kind_hint="airway_management",
            params={
                "procedure": "intubation",
                "stage": "completed",
                "intubated": True,
            },
        ),
        target_vital_names=["O2Sat"],
    )

    assert output["prediction"]["features"]["oxygen_device"] == "vent"
    assert output["prediction"]["features"]["FiO2"] == 1.0
    assert output["prediction"]["features"]["intubated"] is True
    assert output["prediction"]["features"]["vent"] is True


def test_invalid_json_returns_before_vitals_and_records_parse_error() -> None:
    llm = FakeLLMClient("not json")
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"] == BASE_VITALS
    assert output["metadata"]["llm_raw_response"] == "not json"
    assert output["metadata"]["llm_parse_error"] is not None


def test_wrong_adjustments_shape_falls_back_and_records_schema_mismatch() -> None:
    llm = FakeLLMClient('{"adjustments": {"O2Sat": 2}}')
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"] == BASE_VITALS
    assert output["metadata"]["llm_parse_error"] is not None
    assert "vitals" in output["metadata"]["llm_parse_error"]


def test_llm_exception_metadata_is_sanitized() -> None:
    engine = PureLLMEngine(RaisingLLMClient())

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    serialized_metadata = json.dumps(output["metadata"], sort_keys=True)
    assert output["prediction"]["vitals"] == BASE_VITALS
    assert output["metadata"]["llm_raw_response"] == ""
    assert output["metadata"]["llm_parse_error"] == "LLM call failed: RuntimeError"
    assert output["metadata"]["llm_api_error"] == "LLM call failed: RuntimeError"
    assert "secret-api-key" not in serialized_metadata


def test_prompt_omits_forbidden_leak_fields_and_values() -> None:
    llm = FakeLLMClient('{"vitals": {"O2Sat": 88}}')
    engine = PureLLMEngine(llm)
    engine_input = _engine_input()
    engine_input["label"] = {
        "target": {"vitals": {"O2Sat": 99.1234}},
        "leak": "LABEL_LEAK_VALUE",
    }
    engine_input["source"] = {"modifier_text": "SOURCE_MODIFIER_TEXT_LEAK"}
    engine_input["rule_prediction"] = {"vitals": {"O2Sat": 99}}
    engine_input["case_context"]["supporting_findings"] = [
        {"result_summary": "SUPPORTING_FINDINGS_LEAK"}
    ]
    engine_input["before"]["features"]["source"] = "FEATURE_SOURCE_LEAK"
    engine_input["before"]["features"]["target"] = "FEATURE_TARGET_LEAK"
    engine_input["action"]["source"] = "ACTION_SOURCE_LEAK"
    engine_input["action"]["params"]["modifier_text"] = "PARAM_MODIFIER_TEXT_LEAK"
    engine_input["action"]["params"]["supporting_findings"] = "PARAM_FINDINGS_LEAK"
    engine_input["action"]["params"]["rule_prediction"] = "RULE_PREDICTION_LEAK"

    engine.predict(engine_input, target_vital_names=["O2Sat"])

    prompt = llm.prompts[0]
    assert "LABEL_LEAK_VALUE" not in prompt
    assert "SOURCE_MODIFIER_TEXT_LEAK" not in prompt
    assert "RULE_PREDICTION_LEAK" not in prompt
    assert "SUPPORTING_FINDINGS_LEAK" not in prompt
    assert "FEATURE_SOURCE_LEAK" not in prompt
    assert "FEATURE_TARGET_LEAK" not in prompt
    assert "ACTION_SOURCE_LEAK" not in prompt
    assert "PARAM_MODIFIER_TEXT_LEAK" not in prompt
    assert "PARAM_FINDINGS_LEAK" not in prompt
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


def test_prompt_includes_no_action_and_airway_management_guidance() -> None:
    prompt = build_llm_prompt(
        engine_input=_engine_input(kind_hint="no_action", params={"elapsed_min": 5}),
        target_vital_names=["HR", "O2Sat"],
    )

    assert "For no_action:" in prompt
    assert "natural time progression" in prompt
    assert "For airway_management:" in prompt
    assert "Completed successful intubation" in prompt


def test_output_uses_common_engine_output_shape() -> None:
    llm = FakeLLMClient('{"vitals": {"O2Sat": 88}}')
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert set(output) == {"prediction", "metadata"}
    assert tuple(output["prediction"]["vitals"]) == CANONICAL_VITAL_KEYS
    assert output["metadata"] == {
        "engine": "pure_llm",
        "llm_raw_response": '{"vitals": {"O2Sat": 88}}',
        "llm_reasoning": {},
        "llm_parse_error": None,
        "llm_api_error": None,
        "model": "fake-model",
    }


def test_parse_llm_vitals_tolerates_markdown_code_fences_and_direct_dict() -> None:
    fenced = parse_llm_vitals(
        """```json
{"vitals": {"O2Sat": 88, "HR": null, "unknown": 999}}
```"""
    )
    direct = parse_llm_vitals('{"O2Sat": 90, "MAP": 80}')

    assert fenced["O2Sat"] == 88
    assert fenced["HR"] is None
    assert "unknown" not in fenced
    assert direct["O2Sat"] == 90
    assert "MAP" not in direct


def test_parse_llm_vitals_invalid_shapes_return_all_null_vitals() -> None:
    expected = {vital: None for vital in CANONICAL_VITAL_KEYS}

    assert parse_llm_vitals("not json") == expected
    assert parse_llm_vitals("null") == expected
    assert parse_llm_vitals("[]") == expected


def test_llm_predictions_are_clamped_to_physiologic_bounds() -> None:
    llm = FakeLLMClient(
        '{"vitals": {"HR": 999, "BP_sys": 1, "BP_dia": 999, "RR": 99, "O2Sat": 101, "T": 99}}'
    )
    engine = PureLLMEngine(llm)

    output = engine.predict(_engine_input())

    assert output["prediction"]["vitals"] == {
        "HR": 250,
        "BP_sys": 40,
        "BP_dia": 160,
        "RR": 60,
        "O2Sat": 100,
        "T": 43,
    }
