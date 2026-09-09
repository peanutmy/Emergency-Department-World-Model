"""Tests for few-shot example banking, selection, and rendering."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.few_shot import ExampleBank, ExampleSelector
from transition_engines.few_shot import render_hybrid_example, render_pure_example


def _pair(
    *,
    pair_id: str = "p1",
    kind_hint: str = "oxygen_support",
    before_o2: float = 80,
    target_o2: float = 90,
    params: dict | None = None,
) -> dict:
    return {
        "id": pair_id,
        "pair_type": "physiology_response",
        "input": {
            "before": {
                "vitals": {
                    "HR": 110,
                    "BP_sys": 120,
                    "BP_dia": 70,
                    "RR": 30,
                    "O2Sat": before_o2,
                    "T": 37.0,
                },
                "features": {"rhythm": "sinus", "oxygen_device": None},
            },
            "action": {
                "raw_text": "apply NRB",
                "kind_hint": kind_hint,
                "params": params if params is not None else {"oxygen_device": "NRB"},
            },
        },
        "label": {
            "target": {"vitals": {"O2Sat": target_o2}, "features": {}},
            "evaluation": {"target_vitals": ["O2Sat"], "target_features": []},
        },
    }


def _case_doc(case_id: str, pairs: list[dict]) -> dict:
    return {"case_id": case_id, "pairs": pairs}


# ---- renderers ----------------------------------------------------------


def test_render_pure_example_expected_output_uses_stated_target() -> None:
    rendered = render_pure_example(_pair(target_o2=90))
    assert rendered["expected_output"]["vitals"]["O2Sat"] == 90
    assert rendered["before_vitals"]["O2Sat"] == 80
    assert rendered["action"]["kind_hint"] == "oxygen_support"


def test_render_hybrid_example_residual_is_target_minus_rule() -> None:
    # NRB on O2Sat 80 -> rule moves toward 94 with max_delta 8 -> 88.
    # Stated target 90 -> residual = 90 - 88 = +2.
    rendered = render_hybrid_example(_pair(before_o2=80, target_o2=90))
    assert rendered["rule_based_prediction"]["O2Sat"] == 88
    assert rendered["expected_output"]["adjustments"]["O2Sat"] == 2


def test_render_hybrid_example_skips_vitals_without_numeric_target() -> None:
    pair = _pair()
    pair["label"]["target"]["vitals"] = {}  # no numeric target
    rendered = render_hybrid_example(pair)
    assert rendered["expected_output"]["adjustments"] == {}


# ---- bank ---------------------------------------------------------------


def test_bank_from_case_docs_collects_pairs_with_case_id() -> None:
    bank = ExampleBank.from_case_docs(
        [_case_doc("A", [_pair(pair_id="a1")]), _case_doc("B", [_pair(pair_id="b1")])]
    )
    case_ids = {entry["case_id"] for entry in bank.entries}
    assert case_ids == {"A", "B"}


def test_bank_from_dataset_loads_real_transition_pairs() -> None:
    bank = ExampleBank.from_dataset(ROOT / "transitions", recursive=True)
    assert len(bank.entries) > 50
    assert all(entry.get("case_id") for entry in bank.entries)


# ---- selector -----------------------------------------------------------


def _engine_input(case_id: str, kind_hint: str = "oxygen_support") -> dict:
    return {
        "case_id": case_id,
        "action": {"raw_text": "x", "kind_hint": kind_hint, "params": {}},
        "before": {"vitals": {}, "features": {}},
    }


def _mixed_bank() -> ExampleBank:
    return ExampleBank.from_case_docs(
        [
            _case_doc("A", [_pair(pair_id="a1", kind_hint="oxygen_support")]),
            _case_doc("B", [_pair(pair_id="b1", kind_hint="fluid_bolus")]),
            _case_doc("C", [_pair(pair_id="c1", kind_hint="oxygen_support")]),
            _case_doc("D", [_pair(pair_id="d1", kind_hint="vasopressor")]),
        ]
    )


def test_selector_excludes_current_case() -> None:
    selector = ExampleSelector(_mixed_bank(), k=3, strategy="static", seed=0)
    examples = selector.select(_engine_input("A"))
    assert len(examples) == 3
    for example in examples:
        assert example["_case_id"] != "A"


def test_selector_returns_at_most_k() -> None:
    selector = ExampleSelector(_mixed_bank(), k=2, strategy="static", seed=0)
    assert len(selector.select(_engine_input("A"))) == 2


def test_selector_is_deterministic_for_same_seed() -> None:
    a = ExampleSelector(_mixed_bank(), k=3, strategy="static", seed=7)
    b = ExampleSelector(_mixed_bank(), k=3, strategy="static", seed=7)
    assert a.select(_engine_input("A")) == b.select(_engine_input("A"))


def test_kind_hint_matched_prefers_same_kind_first() -> None:
    selector = ExampleSelector(_mixed_bank(), k=1, strategy="kind_hint_matched", seed=0)
    # current case B (fluid_bolus excluded), asking for oxygen_support:
    examples = selector.select(_engine_input("B", kind_hint="oxygen_support"))
    assert examples[0]["_kind_hint"] == "oxygen_support"


def test_static_strategy_ignores_kind_hint_for_ordering() -> None:
    selector = ExampleSelector(_mixed_bank(), k=3, strategy="static", seed=0)
    one = selector.select(_engine_input("A", kind_hint="oxygen_support"))
    two = selector.select(_engine_input("A", kind_hint="vasopressor"))
    assert [e["_case_id"] for e in one] == [e["_case_id"] for e in two]


def test_selector_returns_empty_when_only_current_case_available() -> None:
    bank = ExampleBank.from_case_docs([_case_doc("A", [_pair(), _pair(pair_id="a2")])])
    selector = ExampleSelector(bank, k=3, strategy="static", seed=0)
    assert selector.select(_engine_input("A")) == []


def test_selector_renders_engine_kind_payload() -> None:
    selector = ExampleSelector(
        _mixed_bank(), k=1, strategy="static", seed=0, engine_kind="hybrid"
    )
    example = selector.select(_engine_input("A"))[0]
    assert "rule_based_prediction" in example
    assert "adjustments" in example["expected_output"]

    pure = ExampleSelector(
        _mixed_bank(), k=1, strategy="static", seed=0, engine_kind="pure_llm"
    )
    example = pure.select(_engine_input("A"))[0]
    assert "vitals" in example["expected_output"]
