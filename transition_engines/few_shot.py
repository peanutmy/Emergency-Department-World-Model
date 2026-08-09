"""Few-shot example banking, selection, and rendering for LLM engines.

Examples are drawn leave-one-case-out from the transition dataset: never from
the case currently under prediction. Selection is deterministic given a seed so
that the same demonstrations are shown to every model, keeping cross-model
comparisons fair. Rendering matches each engine's output contract:

* ``pure_llm`` examples show the stated post-action target vitals (absolute).
* ``hybrid`` examples show the rule-based baseline plus the ideal residual
  (target - rule baseline), i.e. exactly the adjustments the model is asked for.

All selection/rendering lives here; the engines stay agnostic and only splice a
pre-rendered list into the prompt when one is supplied.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .common import (
    CANONICAL_VITAL_KEYS,
    direction,
    sanitize_case_context,
)
from .rule_based import RuleBasedEngine


ENGINE_KINDS = ("hybrid", "pure_llm")
STRATEGIES = ("static", "kind_hint_matched")

_FORBIDDEN_KEYS = {
    "label",
    "modifier_text",
    "rule_prediction",
    "source",
    "supporting_findings",
    "target",
}

_RULE_ENGINE = RuleBasedEngine()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_pure_example(pair: dict[str, Any]) -> dict[str, Any]:
    """Render a demonstration for the pure-LLM (absolute vitals) contract."""

    target = _mapping(_mapping(pair.get("label")).get("target")).get("vitals", {})
    expected = {
        vital: target.get(vital)
        for vital in _target_vitals(pair)
        if _is_number(target.get(vital))
    }
    return {
        **_example_context(pair),
        "expected_output": {"vitals": expected},
    }


def render_hybrid_example(pair: dict[str, Any]) -> dict[str, Any]:
    """Render a demonstration for the hybrid (residual adjustment) contract."""

    rule_vitals = _rule_prediction(pair)
    target = _mapping(_mapping(pair.get("label")).get("target")).get("vitals", {})
    adjustments: dict[str, Any] = {}
    for vital in _target_vitals(pair):
        target_value = target.get(vital)
        rule_value = rule_vitals.get(vital)
        if _is_number(target_value) and _is_number(rule_value):
            adjustments[vital] = target_value - rule_value
    return {
        **_example_context(pair),
        "rule_based_prediction": rule_vitals,
        "expected_output": {"adjustments": adjustments},
    }


def render_direction_first_example(entry: dict[str, Any]) -> dict[str, Any]:
    """Render both stages of a direction-first hybrid demonstration."""

    pair = _mapping(entry.get("pair"))
    before = _mapping(_mapping(pair.get("input")).get("before")).get("vitals", {})
    target = _mapping(_mapping(pair.get("label")).get("target")).get("vitals", {})
    directions: dict[str, str] = {}
    magnitudes: dict[str, float | None] = {}

    for vital in _target_vitals(pair):
        if vital not in CANONICAL_VITAL_KEYS:
            continue
        before_value = before.get(vital) if isinstance(before, dict) else None
        target_value = target.get(vital)
        if not _is_number(before_value) or not _is_number(target_value):
            continue
        label = direction(before_value, target_value, vital)
        if label is None:
            continue
        directions[vital] = {
            "up": "increase",
            "down": "decrease",
            "stable": "stable",
        }[label]
        magnitudes[vital] = (
            abs(float(target_value) - float(before_value))
            if label != "stable"
            else None
        )

    case_context = sanitize_case_context(_mapping(entry.get("case_context")))
    return {
        "case": {
            "category": entry.get("category"),
            "scenario_description": entry.get("scenario_description"),
            "case_context": _scrub(deepcopy(case_context)),
        },
        **_example_context(pair),
        "rule_based_prediction": _rule_prediction(pair),
        "stage_1_expected_output": {"directions": directions},
        "stage_2_expected_output": {"magnitudes": magnitudes},
    }


_RENDERERS = {"hybrid": render_hybrid_example, "pure_llm": render_pure_example}


# ---------------------------------------------------------------------------
# Example bank
# ---------------------------------------------------------------------------


class ExampleBank:
    """A flat pool of transition pairs tagged with their owning case id."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries

    @classmethod
    def from_case_docs(cls, case_docs: Iterable[dict[str, Any]]) -> "ExampleBank":
        entries: list[dict[str, Any]] = []
        for case_doc in case_docs:
            case_id = case_doc.get("case_id")
            for pair in case_doc.get("pairs", []):
                if not _has_numeric_target(pair):
                    continue
                entries.append(
                    {
                        "case_id": case_id,
                        "pair_id": pair.get("id"),
                        "category": case_doc.get("category"),
                        "scenario_description": case_doc.get(
                            "scenario_description"
                        ),
                        "case_context": deepcopy(case_doc.get("case_context", {})),
                        "kind_hint": _mapping(_mapping(pair.get("input")).get("action")).get(
                            "kind_hint"
                        ),
                        "pair": pair,
                    }
                )
        return cls(entries)

    @classmethod
    def from_dataset(
        cls, path: str | Path, recursive: bool = True
    ) -> "ExampleBank":
        return cls.from_case_docs(_load_case_docs(path, recursive=recursive))


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class ExampleSelector:
    """Deterministic leave-one-case-out few-shot example selector."""

    def __init__(
        self,
        bank: ExampleBank,
        *,
        k: int = 3,
        strategy: str = "static",
        seed: int = 0,
        engine_kind: str = "hybrid",
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy!r}")
        if engine_kind not in ENGINE_KINDS:
            raise ValueError(f"Unsupported engine_kind: {engine_kind!r}")
        if k < 0:
            raise ValueError("k must be non-negative")

        self.k = k
        self.strategy = strategy
        self.engine_kind = engine_kind
        self._render = _RENDERERS[engine_kind]
        # A single seeded shuffle fixes a stable global order; per-prediction
        # selection then filters/re-ranks that order deterministically.
        self._ordered = list(bank.entries)
        random.Random(seed).shuffle(self._ordered)

    def select(self, engine_input: dict[str, Any]) -> list[dict[str, Any]]:
        if self.k == 0:
            return []
        current_case = engine_input.get("case_id")
        current_kind = _mapping(engine_input.get("action")).get("kind_hint")
        chosen = self._choose(current_case, current_kind)

        rendered: list[dict[str, Any]] = []
        for entry in chosen:
            if entry["case_id"] == current_case:
                raise ValueError(
                    "few-shot leakage: selected example shares the current case_id"
                )
            payload = self._render(entry["pair"])
            rendered.append(
                {
                    "_case_id": entry["case_id"],
                    "_kind_hint": entry["kind_hint"],
                    **payload,
                }
            )
        return rendered

    def _choose(
        self, current_case: Any, current_kind: Any
    ) -> list[dict[str, Any]]:
        candidates = [e for e in self._ordered if e["case_id"] != current_case]
        if self.strategy == "kind_hint_matched":
            matched = [e for e in candidates if e["kind_hint"] == current_kind]
            rest = [e for e in candidates if e["kind_hint"] != current_kind]
            candidates = matched + rest
        return candidates[: self.k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _example_context(pair: dict[str, Any]) -> dict[str, Any]:
    before = _mapping(_mapping(pair.get("input")).get("before"))
    action = _mapping(_mapping(pair.get("input")).get("action"))
    return {
        "before_vitals": _canonical_vitals(before.get("vitals", {})),
        "before_features": _scrub(deepcopy(before.get("features", {}) or {})),
        "action": {
            "raw_text": action.get("raw_text"),
            "kind_hint": action.get("kind_hint"),
            "params": _scrub(deepcopy(action.get("params", {}) or {})),
        },
    }


def _rule_prediction(pair: dict[str, Any]) -> dict[str, Any]:
    pair_input = _mapping(pair.get("input"))
    engine_input = {
        "before": pair_input.get("before", {}),
        "action": pair_input.get("action", {}),
    }
    rule_output = _RULE_ENGINE.predict(engine_input)
    return _canonical_vitals(rule_output.get("prediction", {}).get("vitals", {}))


def _target_vitals(pair: dict[str, Any]) -> list[str]:
    evaluation = _mapping(_mapping(pair.get("label")).get("evaluation"))
    return list(evaluation.get("target_vitals", []))


def _has_numeric_target(pair: dict[str, Any]) -> bool:
    target = _mapping(_mapping(pair.get("label")).get("target")).get("vitals", {})
    return any(_is_number(target.get(vital)) for vital in _target_vitals(pair))


def _load_case_docs(path: str | Path, *, recursive: bool) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        pattern = "**/*.json" if recursive else "*.json"
        files = sorted(path.glob(pattern))

    case_docs: list[dict[str, Any]] = []
    for file in files:
        try:
            with file.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("pairs"), list):
            case_docs.append(doc)
    return case_docs


def _canonical_vitals(vitals: Any) -> dict[str, Any]:
    if not isinstance(vitals, dict):
        vitals = {}
    return {vital: deepcopy(vitals.get(vital)) for vital in CANONICAL_VITAL_KEYS}


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub(nested)
            for key, nested in value.items()
            if key not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_scrub(nested) for nested in value]
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "ENGINE_KINDS",
    "STRATEGIES",
    "ExampleBank",
    "ExampleSelector",
    "render_direction_first_example",
    "render_hybrid_example",
    "render_pure_example",
]
