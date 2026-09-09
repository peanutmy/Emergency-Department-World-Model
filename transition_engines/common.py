"""Shared input builders and evaluation helpers for transition-pair engines."""
from __future__ import annotations

from copy import deepcopy
from math import isclose, sqrt
from statistics import median
from typing import Any, Iterator, Literal, NotRequired, TypedDict


VitalKey = Literal["HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"]
Direction = Literal["up", "down", "stable"]

CANONICAL_VITAL_KEYS: tuple[VitalKey, ...] = (
    "HR",
    "BP_sys",
    "BP_dia",
    "RR",
    "O2Sat",
    "T",
)
CANONICAL_VITAL_KEY_SET = set(CANONICAL_VITAL_KEYS)

DIRECTION_DEAD_ZONES: dict[VitalKey, float] = {
    "HR": 5.0,
    "BP_sys": 5.0,
    "BP_dia": 5.0,
    "RR": 2.0,
    "O2Sat": 2.0,
    "T": 0.2,
}

NORMALIZED_L2_SCALES: dict[VitalKey, float] = {
    "HR": 10.0,
    "BP_sys": 10.0,
    "BP_dia": 10.0,
    "RR": 5.0,
    "O2Sat": 5.0,
    "T": 0.5,
}


class EngineInput(TypedDict):
    case_id: str | None
    category: str | None
    scenario_description: str | None
    case_context: dict[str, Any]
    pair_id: str | None
    before: dict[str, Any]
    action: dict[str, Any]


class EngineOutput(TypedDict):
    prediction: dict[str, dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]


def sanitize_case_context(case_context: dict[str, Any] | None) -> dict[str, Any]:
    """Return a defensive case context copy without engine-forbidden findings."""

    sanitized = deepcopy(case_context) if case_context is not None else {}
    sanitized.pop("supporting_findings", None)
    return sanitized


def build_engine_input(case_doc: dict[str, Any], pair: dict[str, Any]) -> EngineInput:
    """Build the engine-safe input for a single transition pair."""

    pair_input = pair["input"]
    return {
        "case_id": case_doc.get("case_id"),
        "category": case_doc.get("category"),
        "scenario_description": case_doc.get("scenario_description"),
        "case_context": sanitize_case_context(case_doc.get("case_context")),
        "pair_id": pair.get("id"),
        "before": deepcopy(pair_input["before"]),
        "action": deepcopy(pair_input["action"]),
    }


def iter_engine_inputs(
    case_doc: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], EngineInput]]:
    """Yield original pairs with their engine-safe inputs."""

    for pair in case_doc["pairs"]:
        yield pair, build_engine_input(case_doc, pair)


def make_baseline_output_from_before(engine_input: EngineInput) -> EngineOutput:
    """Create a no-change baseline output with every canonical vital key."""

    before = engine_input.get("before", {})
    before_vitals = before.get("vitals", {})
    before_features = before.get("features", {})
    return {
        "prediction": {
            "vitals": {
                vital: deepcopy(before_vitals.get(vital))
                for vital in CANONICAL_VITAL_KEYS
            },
            "features": deepcopy(before_features)
            if isinstance(before_features, dict)
            else {},
        }
    }


def direction(
    before: float | int | None,
    after: float | int | None,
    vital: VitalKey,
) -> Direction | None:
    """Classify a vital change using the configured dead zone."""

    if before is None or after is None:
        return None
    delta = float(after) - float(before)
    abs_delta = abs(delta)
    dead_zone = DIRECTION_DEAD_ZONES[vital]
    if abs_delta <= dead_zone or isclose(abs_delta, dead_zone):
        return "stable"
    if delta > 0:
        return "up"
    return "down"


def evaluate_pair(
    pair: dict[str, Any],
    engine_output: EngineOutput | dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one engine output using only the pair's target vitals."""

    target_vitals = list(pair["label"]["evaluation"].get("target_vitals", []))
    before_vitals = pair["input"]["before"].get("vitals", {})
    target_values = pair["label"].get("target", {}).get("vitals", {})
    predicted_values = engine_output.get("prediction", {}).get("vitals", {})

    direction_match: dict[str, bool] = {}
    missing_predictions: dict[str, bool] = {}
    skipped_vitals: list[str] = []
    normalized_squares: list[float] = []
    num_missing_targets = 0

    for vital in target_vitals:
        predicted_value = predicted_values.get(vital)
        target_value = target_values.get(vital)

        missing_predictions[vital] = predicted_value is None
        if target_value is None:
            num_missing_targets += 1

        if vital not in CANONICAL_VITAL_KEY_SET:
            skipped_vitals.append(vital)
            continue

        before_value = before_vitals.get(vital)
        if target_value is None or predicted_value is None:
            direction_match[vital] = False
            continue

        normalized_squares.append(
            ((float(predicted_value) - float(target_value)) / NORMALIZED_L2_SCALES[vital])
            ** 2
        )

        if before_value is None:
            direction_match[vital] = False
            continue

        true_direction = direction(before_value, target_value, vital)
        pred_direction = direction(before_value, predicted_value, vital)
        direction_match[vital] = pred_direction == true_direction

    direction_accuracy = (
        sum(1 for matched in direction_match.values() if matched) / len(direction_match)
        if direction_match
        else None
    )
    normalized_l2 = sqrt(sum(normalized_squares)) if normalized_squares else None

    return {
        "pair_id": pair.get("id"),
        "target_vitals": target_vitals,
        "skipped_vitals": skipped_vitals,
        "direction_match": direction_match,
        "direction_accuracy": direction_accuracy,
        "normalized_l2": normalized_l2,
        "num_target_vitals": len(target_vitals),
        "num_missing_predictions": sum(
            1 for missing in missing_predictions.values() if missing
        ),
        "num_missing_targets": num_missing_targets,
        "missing_predictions": missing_predictions,
    }


def evaluate_dataset(
    case_doc: dict[str, Any],
    engine_outputs_by_pair_id: dict[str, EngineOutput | dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate all pairs with supplied outputs and aggregate numeric metrics."""

    pair_results = [
        evaluate_pair(pair, engine_outputs_by_pair_id.get(pair.get("id"), {}))
        for pair in case_doc["pairs"]
    ]

    direction_accuracies = [
        result["direction_accuracy"]
        for result in pair_results
        if result["direction_accuracy"] is not None
    ]
    normalized_l2_values = [
        result["normalized_l2"]
        for result in pair_results
        if result["normalized_l2"] is not None
    ]
    evaluated_pairs = [
        result
        for result in pair_results
        if result["direction_accuracy"] is not None or result["normalized_l2"] is not None
    ]

    return {
        "case_id": case_doc.get("case_id"),
        "num_pairs": len(pair_results),
        "pair_results": pair_results,
        "aggregate": {
            "direction_accuracy_mean": _mean_or_none(direction_accuracies),
            "normalized_l2_mean": _mean_or_none(normalized_l2_values),
            "normalized_l2_median": (
                median(normalized_l2_values) if normalized_l2_values else None
            ),
            "num_evaluated_pairs": len(evaluated_pairs),
        },
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "CANONICAL_VITAL_KEYS",
    "CANONICAL_VITAL_KEY_SET",
    "DIRECTION_DEAD_ZONES",
    "EngineInput",
    "EngineOutput",
    "NORMALIZED_L2_SCALES",
    "build_engine_input",
    "direction",
    "evaluate_dataset",
    "evaluate_pair",
    "iter_engine_inputs",
    "make_baseline_output_from_before",
    "sanitize_case_context",
]
