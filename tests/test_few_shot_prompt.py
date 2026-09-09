"""Tests for the additive few-shot ``examples`` parameter on prompt builders.

These guard the ED-world-model invariant: with ``examples=None`` (the default,
and the only path the runtime ever uses) each prompt must be byte-for-byte
identical to the zero-shot prompt. They also verify that supplied examples are
rendered into the prompt ahead of the live ``Case:`` block.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.common import CANONICAL_VITAL_KEYS
from transition_engines import hybrid_engine, pure_llm_engine


BASE_VITALS = {
    "HR": 100,
    "BP_sys": 120,
    "BP_dia": 70,
    "RR": 20,
    "O2Sat": 88,
    "T": 37.0,
}


def _engine_input() -> dict:
    return {
        "case_id": "case-1",
        "category": "Respiratory",
        "scenario_description": "Synthetic safe scenario.",
        "case_context": {"demographics": {"age": 44}, "history": ["sob"]},
        "before": {"vitals": dict(BASE_VITALS), "features": {"rhythm": "sinus"}},
        "action": {
            "raw_text": "apply oxygen",
            "kind_hint": "oxygen_support",
            "params": {"oxygen_device": "NRB"},
        },
    }


SENTINEL = "SENTINEL_EXAMPLE_MARKER_12345"


def test_hybrid_examples_none_matches_zero_shot_prompt() -> None:
    base = hybrid_engine.build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=["O2Sat"],
    )
    with_none = hybrid_engine.build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=["O2Sat"],
        examples=None,
    )
    assert with_none == base
    assert SENTINEL not in with_none


def test_pure_examples_none_matches_zero_shot_prompt() -> None:
    base = pure_llm_engine.build_llm_prompt(
        engine_input=_engine_input(),
        target_vital_names=["O2Sat"],
    )
    with_none = pure_llm_engine.build_llm_prompt(
        engine_input=_engine_input(),
        target_vital_names=["O2Sat"],
        examples=None,
    )
    assert with_none == base
    assert SENTINEL not in with_none


def test_hybrid_examples_render_before_case_block() -> None:
    examples = [{"marker": SENTINEL, "expected_output": {"adjustments": {"O2Sat": 2}}}]
    prompt = hybrid_engine.build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=["O2Sat"],
        examples=examples,
    )
    assert SENTINEL in prompt
    assert prompt.index(SENTINEL) < prompt.index("Case:")


def test_pure_examples_render_before_case_block() -> None:
    examples = [{"marker": SENTINEL, "expected_output": {"vitals": {"O2Sat": 90}}}]
    prompt = pure_llm_engine.build_llm_prompt(
        engine_input=_engine_input(),
        target_vital_names=["O2Sat"],
        examples=examples,
    )
    assert SENTINEL in prompt
    assert prompt.index(SENTINEL) < prompt.index("Case:")


def test_hybrid_predict_forwards_examples_to_prompt() -> None:
    class CapturingClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return '{"adjustments": {"O2Sat": 0}}'

    from transition_engines.rule_based import RuleBasedEngine

    client = CapturingClient()
    engine = hybrid_engine.HybridEngine(RuleBasedEngine(), client)
    engine.predict(
        _engine_input(),
        target_vital_names=["O2Sat"],
        examples=[{"marker": SENTINEL}],
    )
    assert SENTINEL in client.prompts[0]


def test_pure_predict_forwards_examples_to_prompt() -> None:
    class CapturingClient:
        def __init__(self) -> None:
            self.model = "fake"
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return '{"vitals": {}}'

    client = CapturingClient()
    engine = pure_llm_engine.PureLLMEngine(client)
    engine.predict(
        _engine_input(),
        target_vital_names=["O2Sat"],
        examples=[{"marker": SENTINEL}],
    )
    assert SENTINEL in client.prompts[0]


def test_underscore_metadata_keys_are_stripped_from_prompt() -> None:
    examples = [
        {
            "_case_id": "SECRET_META_CASE",
            "_kind_hint": "oxygen_support",
            "before_vitals": {"O2Sat": 80},
            "expected_output": {"adjustments": {"O2Sat": 2}},
        }
    ]
    prompt = hybrid_engine.build_llm_prompt(
        engine_input=_engine_input(),
        rule_prediction_vitals=BASE_VITALS,
        target_vital_names=["O2Sat"],
        examples=examples,
    )
    assert "SECRET_META_CASE" not in prompt
    assert "_case_id" not in prompt
    assert "_kind_hint" not in prompt
    assert "expected_output" in prompt


def test_unused_canonical_keys_constant_is_imported() -> None:
    # Sanity import guard so the module under test stays importable.
    assert len(CANONICAL_VITAL_KEYS) == 6
