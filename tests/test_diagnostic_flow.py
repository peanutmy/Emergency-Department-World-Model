from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.actions.validator as validator_module
import ed_world_model.state.state_manager as state_manager_module
from ed_world_model.actions.registry import ActionRegistry
from ed_world_model.actions.validator import ActionValidator
from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.state.global_state import DiagnosticResult, GlobalState
from ed_world_model.state.state_manager import StateManager


def _state_with_m3_test_bank() -> GlobalState:
    return GlobalState(
        truth_state={
            "test_bank": [
                {
                    "name": "ECG",
                    "result": "LVH and A.fib",
                    "turnaround_turns": 1,
                },
                {
                    "name": "Chest X-ray",
                    "result": "CHF",
                    "turnaround_turns": 3,
                },
                {
                    "name": "Troponin",
                    "result": "Troponin I 40",
                    "turnaround_turns": None,
                },
                {
                    "name": "Point-of-care glucose",
                    "result": "94 mg/dL",
                    "turnaround_turns": 0,
                },
            ]
        },
        runtime_state={"turn_index": 0},
    )


def _validator() -> ActionValidator:
    return ActionValidator(ActionRegistry())


def _payload_contains_text(value: Any, text: str) -> bool:
    if isinstance(value, dict):
        return any(_payload_contains_text(item, text) for item in value.values())
    if isinstance(value, list):
        return any(_payload_contains_text(item, text) for item in value)
    if isinstance(value, str):
        return text in value
    return False


def _payload_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_payload_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_payload_keys(item))
        return keys
    return set()


def test_valid_diagnostic_order_normalizes_name_only_and_rejects_unknown_test() -> None:
    state = _state_with_m3_test_bank()

    result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "ECG"},
        state,
    )

    assert result.ok is True
    assert result.action_type == "diagnostic_order"
    assert result.normalized_action == {
        "type": "diagnostic_order",
        "test_name": "ECG",
    }
    assert set(result.normalized_action) == {"type", "test_name"}
    assert "LVH and A.fib" not in str(result.normalized_action)
    assert "result" not in result.normalized_action
    assert "turnaround_turns" not in result.normalized_action
    assert "raw_text" not in result.normalized_action

    unknown = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "CT head"},
        state,
    )

    assert unknown.ok is False
    assert unknown.normalized_action is None
    assert "Unknown diagnostic test_name 'CT head'." in unknown.errors


def test_validated_diagnostic_orders_create_pending_results_from_test_bank() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    test_bank_before = deepcopy(manager.state.truth_state.test_bank)

    ecg_result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "ECG"},
        manager.state,
    )
    chest_result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "Chest X-ray"},
        manager.state,
    )
    troponin_result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "Troponin"},
        manager.state,
    )

    assert ecg_result.ok is True
    assert chest_result.ok is True
    assert troponin_result.ok is True

    ecg_pending = manager.create_pending_diagnostic_result(
        ecg_result.normalized_action["test_name"]
    )
    chest_pending = manager.create_pending_diagnostic_result(
        chest_result.normalized_action["test_name"]
    )
    troponin_pending = manager.create_pending_diagnostic_result(
        troponin_result.normalized_action["test_name"]
    )

    assert ecg_pending.ready_at_turn == 1
    assert chest_pending.ready_at_turn == 3
    assert troponin_pending.ready_at_turn == DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
    assert manager.state.truth_state.test_bank == test_bank_before
    assert manager.state.known_facts.available_results == []
    assert manager.state.runtime_state.newly_available_results == []


def test_release_ready_diagnostic_results_releases_only_currently_ready_results() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    state_before_release = deepcopy(manager.state.model_dump())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)

    turn_zero_released = manager.release_ready_diagnostic_results(current_turn=0)

    assert turn_zero_released == []
    assert manager.state.known_facts.available_results == []
    assert manager.state.runtime_state.newly_available_results == []

    patient_before_turn_one = deepcopy(manager.state.patient_state.model_dump())
    emotion_before_turn_one = deepcopy(manager.state.psych_state.model_dump())
    test_bank_before_turn_one = deepcopy(manager.state.truth_state.test_bank)

    turn_one_released = manager.release_ready_diagnostic_results(current_turn=1)

    assert turn_one_released == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]
    assert manager.state.known_facts.available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]
    assert manager.state.runtime_state.newly_available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]
    assert all(
        result.name != "Chest X-ray"
        for result in manager.state.known_facts.available_results
    )
    assert manager.state.runtime_state.pending_diagnostic_results == [
        pending
        for pending in manager.state.runtime_state.pending_diagnostic_results
        if pending.test_name == "Chest X-ray"
    ]
    assert manager.state.runtime_state.pending_diagnostic_results[0].ready_at_turn == 3
    assert manager.state.truth_state.test_bank == test_bank_before_turn_one
    assert manager.state.patient_state.model_dump() == patient_before_turn_one
    assert manager.state.psych_state.model_dump() == emotion_before_turn_one

    event_types = [event.type for event in manager.state.runtime_state.current_turn_events]
    assert "diagnostic_release" in event_types
    assert "nurse_shadow_execution" not in event_types
    assert "physiology_update" not in event_types
    assert "emotion_update" not in event_types
    assert manager.state.runtime_state.messages == []
    for event in manager.state.runtime_state.current_turn_events:
        assert "raw_text" not in _payload_keys(event.payload)
    assert manager.state.model_dump() != state_before_release


def test_zero_turnaround_diagnostic_can_release_same_turn() -> None:
    manager = StateManager(_state_with_m3_test_bank())

    pending = manager.create_pending_diagnostic_result(
        "Point-of-care glucose",
        current_turn=0,
    )
    released = manager.release_ready_diagnostic_results(current_turn=0)

    assert pending.ready_at_turn == 0
    assert released == [DiagnosticResult(name="Point-of-care glucose", result="94 mg/dL")]
    assert manager.state.known_facts.available_results == released
    assert manager.state.runtime_state.newly_available_results == released


def test_release_ready_diagnostic_results_with_no_pending_is_noop() -> None:
    manager = StateManager(_state_with_m3_test_bank())

    released = manager.release_ready_diagnostic_results(current_turn=0)

    assert released == []
    assert manager.state.known_facts.available_results == []
    assert manager.state.runtime_state.newly_available_results == []
    assert [
        event
        for event in manager.state.runtime_state.current_turn_events
        if event.type == "diagnostic_release"
    ] == []


def test_releasing_twice_at_same_turn_is_idempotent_after_first_release() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    first_release = manager.release_ready_diagnostic_results(current_turn=1)
    known_after_first = list(manager.state.known_facts.available_results)
    newly_after_first = list(manager.state.runtime_state.newly_available_results)
    release_events_after_first = [
        event
        for event in manager.state.runtime_state.current_turn_events
        if event.type == "diagnostic_release"
    ]

    second_release = manager.release_ready_diagnostic_results(current_turn=1)

    assert first_release == [DiagnosticResult(name="ECG", result="LVH and A.fib")]
    assert second_release == []
    assert manager.state.known_facts.available_results == known_after_first
    assert manager.state.runtime_state.newly_available_results == newly_after_first
    assert [
        event
        for event in manager.state.runtime_state.current_turn_events
        if event.type == "diagnostic_release"
    ] == release_events_after_first


def test_unreleased_result_string_stays_out_of_known_results_and_event_payloads() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)
    manager.advance_turn()

    released = manager.release_ready_diagnostic_results(current_turn=1)

    assert released == []
    assert all(
        result.result != "CHF" for result in manager.state.known_facts.available_results
    )
    assert all(
        result.result != "CHF"
        for result in manager.state.runtime_state.newly_available_results
    )
    for event in (
        manager.state.runtime_state.current_turn_events
        + manager.state.runtime_state.last_turn_events
    ):
        assert not _payload_contains_text(event.payload, "CHF")


def test_diagnostic_release_event_records_current_turn_payload() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)

    manager.release_ready_diagnostic_results(current_turn=1)

    release_events = [
        event
        for event in manager.state.runtime_state.current_turn_events
        if event.type == "diagnostic_release"
    ]
    assert len(release_events) == 1
    assert release_events[0].turn_index == 1
    assert release_events[0].payload == {
        "test_name": "ECG",
        "ordered_at_turn": 0,
        "ready_at_turn": 1,
        "released_at_turn": 1,
    }


def test_advance_turn_clears_newly_available_results_but_keeps_known_results() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)
    manager.release_ready_diagnostic_results(current_turn=1)

    manager.advance_turn()

    assert manager.state.runtime_state.newly_available_results == []
    assert manager.state.known_facts.available_results == [
        DiagnosticResult(name="ECG", result="LVH and A.fib")
    ]
    assert manager.state.runtime_state.last_turn_events[-1].type == "diagnostic_release"
    assert manager.state.runtime_state.current_turn_events == []


def test_release_ready_diagnostic_results_can_release_multiple_results_at_once() -> None:
    manager = StateManager(_state_with_m3_test_bank())
    manager.create_pending_diagnostic_result("ECG", current_turn=0)
    manager.create_pending_diagnostic_result("Troponin", current_turn=0)
    manager.create_pending_diagnostic_result("Chest X-ray", current_turn=0)

    released = manager.release_ready_diagnostic_results(current_turn=3)

    assert released == [
        DiagnosticResult(name="ECG", result="LVH and A.fib"),
        DiagnosticResult(name="Troponin", result="Troponin I 40"),
        DiagnosticResult(name="Chest X-ray", result="CHF"),
    ]
    assert manager.state.known_facts.available_results == released
    assert manager.state.runtime_state.newly_available_results == released
    assert manager.state.runtime_state.pending_diagnostic_results == []


def test_diagnostic_flow_does_not_import_transition_pairs_or_engines() -> None:
    validator_source = inspect.getsource(validator_module)
    state_manager_source = inspect.getsource(state_manager_module)

    for source in (validator_source, state_manager_source):
        assert "transition_engines" not in source
        assert "transitions/" not in source
        assert "pair_label" not in source
        assert "pair_type" not in source
