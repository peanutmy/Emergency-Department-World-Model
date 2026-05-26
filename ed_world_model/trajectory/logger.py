"""Lightweight JSON-only trajectory logger."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TRAJECTORY_FILENAME = "trajectory.json"
SUMMARY_FILENAME = "summary.json"
REDACTED_VALUE = "[REDACTED]"

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "openai_api_key",
    "authorization",
    "access_token",
    "bearer_token",
)

_PATIENT_STATE_KEYS = ("vitals", "features", "status_flags")
_KNOWN_FACTS_KEYS = (
    "chief_complaint",
    "known_history",
    "known_allergies",
    "known_medications",
    "known_symptoms",
    "available_results",
)


class TrajectoryLogger:
    """Write existing trajectory records to JSON files.

    The logger does not own state and does not call runtime components. It only
    serializes already-produced turn records.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        trajectory: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.output_dir / TRAJECTORY_FILENAME
        payload = _build_trajectory_payload(trajectory, metadata=metadata)
        _write_json(path, payload)
        return path

    def save_summary(self, metadata: Mapping[str, Any]) -> Path:
        path = self.output_dir / SUMMARY_FILENAME
        payload = _build_summary_payload(metadata=metadata)
        _write_json(path, payload)
        return path

    def save_all(
        self,
        trajectory: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        trajectory_payload = _build_trajectory_payload(trajectory, metadata=metadata)
        summary_payload = _build_summary_payload(
            metadata=metadata,
            trajectory_payload=trajectory_payload,
        )
        trajectory_path = self.output_dir / TRAJECTORY_FILENAME
        summary_path = self.output_dir / SUMMARY_FILENAME
        _write_json(trajectory_path, trajectory_payload)
        _write_json(summary_path, summary_payload)
        return {"trajectory": trajectory_path, "summary": summary_path}


def _build_trajectory_payload(
    trajectory: Any,
    *,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = _to_plain(trajectory)
    safe_metadata = _to_plain(metadata or {})

    if isinstance(source, Mapping):
        source_mapping = dict(source)
        raw_turns = source_mapping.get("turns", [])
    elif _is_sequence(source):
        source_mapping = {}
        raw_turns = source
    else:
        raise TypeError("trajectory must be a mapping, sequence, or serializable object.")

    turns = [_normalize_turn(turn) for turn in list(raw_turns or [])]
    final_turn_index = _first_present(
        safe_metadata,
        source_mapping,
        "final_turn_index",
    )
    if final_turn_index is None and turns:
        final_turn_index = turns[-1].get("turn_index_after")
        if final_turn_index is None:
            final_turn_index = turns[-1].get("turn_index")

    payload = {
        "scenario_identifier": _first_present(
            safe_metadata,
            source_mapping,
            "scenario_identifier",
            "scenario_id",
            "scenario_name",
            "name",
        ),
        "scenario_id": _first_present(
            safe_metadata,
            source_mapping,
            "scenario_id",
            "case_id",
            "id",
        ),
        "scenario_name": _first_present(
            safe_metadata,
            source_mapping,
            "scenario_name",
            "name",
            "title",
        ),
        "agent_mode": _first_present(safe_metadata, source_mapping, "agent_mode"),
        "physiology_mode": _first_present(
            safe_metadata,
            source_mapping,
            "physiology_mode",
        ),
        "requested_turns": _first_present(
            safe_metadata,
            source_mapping,
            "requested_turns",
        ),
        "turn_count": len(turns),
        "final_turn_index": final_turn_index,
        "turns": turns,
    }
    return payload


def _build_summary_payload(
    metadata: Mapping[str, Any] | None,
    *,
    trajectory_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_metadata = _to_plain(metadata or {})
    trajectory = dict(trajectory_payload or {})
    turns = list(trajectory.get("turns") or [])

    total_validation_drops = _first_present(
        safe_metadata,
        {},
        "total_validation_drops",
    )
    if total_validation_drops is None:
        total_validation_drops = sum(
            len(turn.get("validation_drops") or []) for turn in turns
        )

    total_parser_errors = _first_present(safe_metadata, {}, "total_parser_errors")
    if total_parser_errors is None:
        total_parser_errors = sum(
            len(turn.get("parser_errors") or []) for turn in turns
        )

    payload = {
        "scenario_identifier": _first_present(
            safe_metadata,
            trajectory,
            "scenario_identifier",
            "scenario_id",
            "scenario_name",
            "name",
        ),
        "scenario_id": _first_present(
            safe_metadata,
            trajectory,
            "scenario_id",
            "case_id",
            "id",
        ),
        "scenario_name": _first_present(
            safe_metadata,
            trajectory,
            "scenario_name",
            "name",
            "title",
        ),
        "agent_mode": _first_present(safe_metadata, trajectory, "agent_mode"),
        "physiology_mode": _first_present(
            safe_metadata,
            trajectory,
            "physiology_mode",
        ),
        "requested_turns": _first_present(
            safe_metadata,
            trajectory,
            "requested_turns",
        ),
        "final_turn_index": _first_present(
            safe_metadata,
            trajectory,
            "final_turn_index",
        ),
        "turns_recorded": _first_present(
            safe_metadata,
            trajectory,
            "turns_recorded",
            "turn_count",
        )
        if not turns
        else len(turns),
        "total_validation_drops": total_validation_drops,
        "total_parser_errors": total_parser_errors,
        "output_timestamp": safe_metadata.get(
            "output_timestamp",
            datetime.now(timezone.utc).isoformat(),
        ),
    }

    for key in (
        "mode_fallback",
        "fallback",
        "fallback_reason",
        "agent_fallback",
        "physiology_fallback",
    ):
        if key in safe_metadata:
            payload[key] = deepcopy(safe_metadata[key])
    return payload


def _normalize_turn(turn: Any) -> dict[str, Any]:
    data = _to_plain(turn)
    if not isinstance(data, Mapping):
        raise TypeError("each trajectory turn must serialize to an object.")
    turn_data = dict(data)
    events = list(turn_data.get("events") or [])
    validation_drops = turn_data.get("validation_drops")
    if validation_drops is None:
        validation_drops = [
            event for event in events if event.get("type") == "validation_drop"
        ]
    parser_errors = turn_data.get("parser_errors")
    if parser_errors is None:
        parser_errors = [
            drop
            for drop in validation_drops
            if drop.get("payload", {}).get("item") == "agent_proposal"
        ]

    return {
        "turn_index": _first_present(turn_data, {}, "turn_index", "turn_index_before"),
        "turn_index_after": turn_data.get("turn_index_after"),
        "state_before": _normalize_state_snapshot(turn_data.get("state_before")),
        "active_agents": list(
            turn_data.get("turn_start_active_agents")
            or turn_data.get("active_agents")
            or []
        ),
        "committed_messages": deepcopy(
            turn_data.get("committed_messages")
            or turn_data.get("messages")
            or []
        ),
        "clinician_action": _normalize_clinician_action(
            turn_data.get("clinician_action")
        ),
        "diagnostic_orders_created": _event_payloads_or_existing(
            turn_data,
            events,
            existing_key="diagnostic_orders_created",
            event_type="diagnostic_order_created",
        ),
        "diagnostic_results_released": deepcopy(
            turn_data.get("diagnostic_results_released")
            or turn_data.get("released_diagnostics")
            or []
        ),
        "nurse_shadow_execution": _event_payloads_or_existing(
            turn_data,
            events,
            existing_key="nurse_shadow_execution",
            event_type="nurse_shadow_execution",
        ),
        "nurse_bedside_verbal_slots": _event_payloads_or_existing(
            turn_data,
            events,
            existing_key="nurse_bedside_verbal_slots",
            alternate_existing_key="nurse_bedside_slots",
            event_type="nurse_bedside_verbal_slot",
        ),
        "physiology_action": _normalize_physiology_action(turn_data),
        "state_after": _normalize_state_snapshot(turn_data.get("state_after")),
        "validation_drops": deepcopy(validation_drops),
        "parser_errors": deepcopy(parser_errors),
        "events": deepcopy(events),
        "terminated": bool(turn_data.get("terminated", False)),
        "termination_reason": turn_data.get("termination_reason"),
    }


def _normalize_state_snapshot(snapshot: Any) -> dict[str, Any]:
    data = _to_plain(snapshot or {})
    if not isinstance(data, Mapping):
        data = {}

    patient_state = data.get("patient_state") or {}
    if not isinstance(patient_state, Mapping):
        patient_state = {}
    known_facts = data.get("known_facts") or {}
    if not isinstance(known_facts, Mapping):
        known_facts = {}

    return {
        "patient_state": {
            key: deepcopy(patient_state.get(key) or {})
            for key in _PATIENT_STATE_KEYS
        },
        "known_facts": {
            key: deepcopy(known_facts.get(key))
            for key in _KNOWN_FACTS_KEYS
            if key in known_facts or key != "chief_complaint"
        },
    }


def _event_payloads_or_existing(
    turn_data: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    existing_key: str,
    event_type: str,
    alternate_existing_key: str | None = None,
) -> list[dict[str, Any]]:
    existing = turn_data.get(existing_key)
    if existing is None and alternate_existing_key is not None:
        existing = turn_data.get(alternate_existing_key)
    if existing is not None:
        return deepcopy(list(existing))
    return [
        deepcopy(event.get("payload") or {})
        for event in events
        if event.get("type") == event_type
    ]


def _normalize_physiology_action(turn_data: Mapping[str, Any]) -> dict[str, Any] | None:
    action = deepcopy(turn_data.get("physiology_action"))
    if action is None:
        kind_hint = turn_data.get("physiology_action_kind_hint")
        if kind_hint is None:
            return None
        action = {"kind_hint": kind_hint}
    if not isinstance(action, Mapping):
        return {"value": action}
    normalized = dict(action)
    normalized.setdefault("raw_text", None)
    normalized.setdefault("params", None)
    return normalized


def _normalize_clinician_action(action: Any) -> dict[str, Any] | None:
    data = _to_plain(action)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        return {"value": deepcopy(data)}

    action_data = dict(data)
    normalized_action = action_data.get("normalized_action")
    if not isinstance(normalized_action, Mapping):
        normalized_action = {}

    action_type = action_data.get("type") or action_data.get("action_type")
    if action_type == "medical_treatment_order":
        result: dict[str, Any] = {
            "type": "medical_treatment_order",
            "action_type": "medical_treatment_order",
        }
        family = action_data.get("family")
        if family is not None:
            result["family"] = deepcopy(family)
        kind_hint = action_data.get("kind_hint") or normalized_action.get("kind_hint")
        if kind_hint is not None:
            result["kind_hint"] = deepcopy(kind_hint)
        if "params" in action_data:
            result["params"] = deepcopy(action_data.get("params"))
        else:
            result["params"] = deepcopy(normalized_action.get("params"))
        return result

    if action_type == "diagnostic_order":
        result = {
            "type": "diagnostic_order",
            "action_type": "diagnostic_order",
        }
        test_name = action_data.get("test_name") or normalized_action.get("test_name")
        if test_name is not None:
            result["test_name"] = deepcopy(test_name)
        return result

    result = {
        key: deepcopy(value)
        for key, value in action_data.items()
        if key != "normalized_action"
    }
    return result or None


def _to_plain(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _to_plain(value.as_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_plain(value.model_dump())
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_sensitive_key(text_key):
                result[text_key] = REDACTED_VALUE
            else:
                result[text_key] = _to_plain(item)
        return result
    if _is_sequence(value):
        return [_to_plain(item) for item in value]
    return deepcopy(value)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in primary and primary[key] is not None:
            return deepcopy(primary[key])
        if key in secondary and secondary[key] is not None:
            return deepcopy(secondary[key])
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["TrajectoryLogger"]
