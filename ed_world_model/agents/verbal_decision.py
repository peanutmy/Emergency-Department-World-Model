"""Strict parsing for transient verbal decisions."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ed_world_model.agents.schemas import VerbalDecision, VerbalDecisionSpeaker


FORBIDDEN_VERBAL_DECISION_KEYS = {
    "raw_text",
    "chain_of_thought",
    "internal_reasoning",
    "action",
    "intent_type",
}


class VerbalDecisionParserError(ValueError):
    """Raised when a transient VerbalDecision response is invalid."""


def parse_verbal_decision(
    output: str | Mapping[str, Any],
    *,
    speaker: VerbalDecisionSpeaker,
) -> VerbalDecision:
    """Parse strict JSON into a transient VerbalDecision."""

    data = _coerce_strict_json_object(output)
    forbidden_key_path = _find_forbidden_key(data)
    if forbidden_key_path is not None:
        raise VerbalDecisionParserError(
            "VerbalDecision must not include raw_text, chain_of_thought, "
            "internal_reasoning, action, or intent_type fields: "
            f"{forbidden_key_path}"
        )
    try:
        decision = VerbalDecision.model_validate(data)
    except ValidationError as exc:
        raise VerbalDecisionParserError(
            f"Invalid VerbalDecision shape: {exc}"
        ) from exc
    if decision.speaker != speaker:
        raise VerbalDecisionParserError(
            f"VerbalDecision speaker must be {speaker!r}."
        )
    return decision


def _coerce_strict_json_object(output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            data = json.loads(
                output,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise VerbalDecisionParserError(
                "VerbalDecision response must be valid strict JSON."
            ) from exc
    elif isinstance(output, Mapping):
        data = dict(output)
    else:
        raise VerbalDecisionParserError(
            "VerbalDecision response must be a JSON object."
        )

    if not isinstance(data, dict):
        raise VerbalDecisionParserError(
            "VerbalDecision response must be a JSON object."
        )
    return data


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value!r}.")


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_VERBAL_DECISION_KEYS:
                return current_path
            nested = _find_forbidden_key(item, current_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_key(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


__all__ = [
    "FORBIDDEN_VERBAL_DECISION_KEYS",
    "VerbalDecisionParserError",
    "parse_verbal_decision",
]
