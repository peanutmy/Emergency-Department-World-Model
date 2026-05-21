from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.common import CANONICAL_VITAL_KEYS
from transition_engines.hybrid_engine import (
    build_llm_prompt,
    HybridEngine,
    merge_vitals,
    parse_llm_adjustments,
)
from transition_engines.rule_based import RuleBasedEngine


BASE_VITALS = {
    "HR": 100,
    "BP_sys": 120,
    "BP_dia": 70,
    "RR": 20,
    "O2Sat": 88,
    "T": 37.0,
}


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RaisingLLMClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        raise self.exc


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


def test_llm_adjustment_is_added_to_rule_prediction() -> None:
    llm = FakeLLMClient(
        '{"adjustments": {"O2Sat": 2}, '
        '"reasoning": {"O2Sat": "Oxygen support should improve saturation."}}'
    )
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(_engine_input())

    rule_vitals = output["metadata"]["rule_prediction"]["vitals"]
    assert rule_vitals["O2Sat"] == 94
    assert output["prediction"]["vitals"]["O2Sat"] == 96
    assert output["prediction"]["features"]["oxygen_device"] == "NRB"
    assert output["metadata"]["rule_prediction"]["features"] == output["prediction"][
        "features"
    ]
    assert output["metadata"]["llm_adjustments"]["O2Sat"] == 2
    assert output["metadata"]["llm_reasoning"]["O2Sat"] == (
        "Oxygen support should improve saturation."
    )
    assert output["metadata"]["llm_error"] is None


def test_target_vital_names_ignore_non_target_adjustments() -> None:
    llm = FakeLLMClient('{"adjustments": {"HR": 20, "O2Sat": 2}}')
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"]["HR"] == 100
    assert output["prediction"]["vitals"]["O2Sat"] == 96


def test_hybrid_feature_prediction_comes_from_rule_engine() -> None:
    llm = FakeLLMClient(
        '{"adjustments": {"O2Sat": 2}, "features": {"oxygen_device": "room_air"}}'
    )
    engine = HybridEngine(RuleBasedEngine(), llm)

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


def test_invalid_json_leaves_rule_prediction_unchanged() -> None:
    engine_input = _engine_input()
    rule_output = RuleBasedEngine().predict(engine_input)
    llm = FakeLLMClient("not json")
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(engine_input)

    assert output["prediction"]["vitals"] == rule_output["prediction"]["vitals"]
    assert output["metadata"]["llm_adjustments"] == {}
    assert output["metadata"]["llm_reasoning"] == {}
    assert output["metadata"]["llm_error"] is None


def test_llm_exception_leaves_rule_prediction_unchanged_and_records_error() -> None:
    engine_input = _engine_input()
    rule_output = RuleBasedEngine().predict(engine_input)
    llm = RaisingLLMClient(RuntimeError("LLM unavailable"))
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(engine_input)

    assert output["prediction"]["vitals"] == rule_output["prediction"]["vitals"]
    assert output["metadata"]["llm_adjustments"] == {}
    assert output["metadata"]["llm_error"] == "RuntimeError('LLM unavailable')"


def test_prompt_omits_forbidden_leak_fields_and_values() -> None:
    llm = FakeLLMClient('{"adjustments": {"O2Sat": 0}}')
    engine = HybridEngine(RuleBasedEngine(), llm)
    engine_input = _engine_input()
    engine_input["label"] = {
        "target": {"vitals": {"O2Sat": 100}},
        "leak": "LABEL_LEAK_VALUE",
    }
    engine_input["source"] = {
        "modifier_text": "SOURCE_MODIFIER_TEXT_LEAK",
    }
    engine_input["case_context"]["supporting_findings"] = [
        {"result_summary": "SUPPORTING_FINDINGS_LEAK"}
    ]
    engine_input["before"]["features"]["source"] = "FEATURE_SOURCE_LEAK"
    engine_input["before"]["features"]["target"] = "FEATURE_TARGET_LEAK"
    engine_input["action"]["source"] = "ACTION_SOURCE_LEAK"
    engine_input["action"]["params"]["modifier_text"] = "PARAM_MODIFIER_TEXT_LEAK"
    engine_input["action"]["params"]["supporting_findings"] = "PARAM_FINDINGS_LEAK"

    engine.predict(engine_input, target_vital_names=["O2Sat"])

    prompt = llm.prompts[0]
    assert "LABEL_LEAK_VALUE" not in prompt
    assert "SOURCE_MODIFIER_TEXT_LEAK" not in prompt
    assert "SUPPORTING_FINDINGS_LEAK" not in prompt
    assert "FEATURE_SOURCE_LEAK" not in prompt
    assert "FEATURE_TARGET_LEAK" not in prompt
    assert "ACTION_SOURCE_LEAK" not in prompt
    assert "PARAM_MODIFIER_TEXT_LEAK" not in prompt
    assert "PARAM_FINDINGS_LEAK" not in prompt
    for forbidden_key in (
        "label",
        "target",
        "source",
        "modifier_text",
        "supporting_findings",
    ):
        assert forbidden_key not in prompt


def test_output_uses_hybrid_engine_output_shape() -> None:
    llm = FakeLLMClient('{"adjustments": {"O2Sat": 2}}')
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert set(output) == {"prediction", "metadata"}
    assert tuple(output["prediction"]["vitals"]) == CANONICAL_VITAL_KEYS
    assert output["metadata"]["engine"] == "hybrid"
    assert tuple(output["metadata"]["rule_prediction"]["vitals"]) == CANONICAL_VITAL_KEYS
    assert output["metadata"]["llm_adjustments"]["O2Sat"] == 2
    assert output["metadata"]["llm_reasoning"] == {}
    assert output["metadata"]["llm_error"] is None


def test_target_vital_names_none_allows_all_canonical_adjustments() -> None:
    llm = FakeLLMClient(
        """{
          "adjustments": {
            "HR": 1,
            "BP_sys": 2,
            "BP_dia": 3,
            "RR": 4,
            "O2Sat": 5,
            "T": 0.5
          }
        }"""
    )
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(_engine_input(kind_hint="not_a_rule"))

    assert output["prediction"]["vitals"] == {
        "HR": 101,
        "BP_sys": 122,
        "BP_dia": 73,
        "RR": 24,
        "O2Sat": 93,
        "T": 37.5,
    }


def test_target_vital_names_empty_allows_no_adjustments() -> None:
    engine_input = _engine_input(kind_hint="not_a_rule")
    rule_output = RuleBasedEngine().predict(engine_input)
    llm = FakeLLMClient('{"adjustments": {"HR": 20, "O2Sat": 5}}')
    engine = HybridEngine(RuleBasedEngine(), llm)

    output = engine.predict(engine_input, target_vital_names=[])

    assert output["prediction"]["vitals"] == rule_output["prediction"]["vitals"]


def test_prompt_template_includes_all_vitals_when_target_vital_names_is_none() -> None:
    prompt = build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=list(CANONICAL_VITAL_KEYS),
    )

    template = _prompt_template(prompt)
    for vital in CANONICAL_VITAL_KEYS:
        assert f'"{vital}": null' in template


def test_prompt_template_only_includes_requested_vitals() -> None:
    prompt = build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=["O2Sat"],
    )

    template = _prompt_template(prompt)
    assert '"O2Sat": null' in template
    for vital in set(CANONICAL_VITAL_KEYS) - {"O2Sat"}:
        assert f'"{vital}": null' not in template


def test_prompt_template_is_empty_when_target_vital_names_is_empty() -> None:
    prompt = build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=[],
    )

    assert _prompt_template(prompt) == '{"adjustments": {}, "reasoning": {}}'


def test_parse_llm_adjustments_tolerates_markdown_code_fences() -> None:
    adjustments = parse_llm_adjustments(
        """```json
{"adjustments": {"O2Sat": 2, "HR": null, "unknown": 999}}
```"""
    )

    assert adjustments["O2Sat"] == 2
    assert adjustments["HR"] is None
    assert "unknown" not in adjustments


def test_merge_uses_before_when_rule_vital_missing_and_clamps() -> None:
    output = merge_vitals(
        rule_prediction_vitals={"HR": 245, "O2Sat": None},
        before_vitals={"HR": 100, "O2Sat": 99},
        adjustments={"HR": 10, "O2Sat": 5},
    )

    assert output["HR"] == 250
    assert output["O2Sat"] == 100
    assert tuple(output) == CANONICAL_VITAL_KEYS


def test_boolean_adjustment_is_ignored() -> None:
    adjustments = parse_llm_adjustments('{"adjustments": {"HR": true}}')
    output = merge_vitals(
        rule_prediction_vitals=BASE_VITALS,
        before_vitals=BASE_VITALS,
        adjustments=adjustments,
    )

    assert adjustments["HR"] is None
    assert output["HR"] == BASE_VITALS["HR"]


def test_nan_and_infinity_adjustments_are_ignored() -> None:
    adjustments = parse_llm_adjustments(
        '{"adjustments": {"HR": NaN, "RR": Infinity, "O2Sat": -Infinity}}'
    )
    output = merge_vitals(
        rule_prediction_vitals=BASE_VITALS,
        before_vitals=BASE_VITALS,
        adjustments=adjustments,
    )

    assert adjustments["HR"] is None
    assert adjustments["RR"] is None
    assert adjustments["O2Sat"] is None
    assert output["HR"] == BASE_VITALS["HR"]
    assert output["RR"] == BASE_VITALS["RR"]
    assert output["O2Sat"] == BASE_VITALS["O2Sat"]


def test_missing_and_null_adjustments_keep_rule_prediction_unchanged() -> None:
    adjustments = parse_llm_adjustments('{"adjustments": {"HR": null}}')
    output = merge_vitals(
        rule_prediction_vitals=BASE_VITALS,
        before_vitals=BASE_VITALS,
        adjustments=adjustments,
    )

    assert adjustments["HR"] is None
    assert adjustments["O2Sat"] is None
    assert output == BASE_VITALS


def test_non_canonical_adjustment_keys_are_ignored() -> None:
    adjustments = parse_llm_adjustments('{"adjustments": {"MAP": 20, "O2Sat": 2}}')
    output = merge_vitals(
        rule_prediction_vitals=BASE_VITALS,
        before_vitals=BASE_VITALS,
        adjustments=adjustments,
    )

    assert "MAP" not in adjustments
    assert output["O2Sat"] == BASE_VITALS["O2Sat"] + 2
    assert set(output) == set(CANONICAL_VITAL_KEYS)


@pytest.mark.parametrize(
    "text",
    ["", "not json", '{"adjustments": {"O2Sat": "2"}}'],
)
def test_parse_llm_adjustments_rejects_invalid_or_non_numeric_values(
    text: str,
) -> None:
    adjustments = parse_llm_adjustments(text)

    if text.startswith("{"):
        assert adjustments["O2Sat"] is None
    else:
        assert adjustments == {}


def _prompt_template(prompt: str) -> str:
    return prompt.rsplit("Return JSON only:", maxsplit=1)[1].strip()
