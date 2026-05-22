from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import inspect
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.actions.validator as validator_module
from ed_world_model.actions.registry import ActionFamily, ActionRegistry, KindHint
from ed_world_model.actions.validator import ActionValidator, ValidationResult
from ed_world_model.state.global_state import GlobalState
from ed_world_model.state.global_state import TestBankItem as DiagnosticTestBankItem


def _validator() -> ActionValidator:
    return ActionValidator(ActionRegistry())


def _state_with_tests() -> GlobalState:
    return GlobalState(
        truth_state={
            "test_bank": [
                {
                    "name": "ECG",
                    "result": "atrial fibrillation",
                    "turnaround_turns": 1,
                },
                {
                    "name": "Chest X-ray",
                    "result": "pulmonary edema",
                    "turnaround_turns": 2,
                },
            ]
        }
    )


def _valid_oxygen_order(params: dict | None = None) -> dict:
    action = {
        "type": "medical_treatment_order",
        "family": ActionFamily.RESPIRATORY_SUPPORT,
        "kind_hint": KindHint.OXYGEN_SUPPORT,
    }
    if params is not None:
        action["params"] = params
    return action


def test_none_clinician_action_is_valid_no_action_without_no_action_engine_action() -> None:
    result = _validator().validate_clinician_proposal(None, GlobalState())

    assert result.ok is True
    assert result.action_type is None
    assert result.normalized_action is None
    assert result.errors == []


def test_valid_medical_treatment_order_passes() -> None:
    result = _validator().validate_clinician_proposal(
        _valid_oxygen_order({"oxygen_device": "NRB", "FiO2": 1.0}),
        GlobalState(),
    )

    assert result.ok is True
    assert result.action_type == "medical_treatment_order"
    assert result.normalized_action["kind_hint"] == KindHint.OXYGEN_SUPPORT
    assert result.normalized_action["raw_text"] is None
    assert result.normalized_action["params"]["oxygen_device"] == "NRB"
    assert result.normalized_action["params"]["FiO2"] == 1.0


def test_valid_medical_treatment_order_normalizes_missing_template_fields() -> None:
    result = _validator().validate_medical_treatment_order(
        _valid_oxygen_order({"oxygen_device": None})
    )

    assert result.ok is True
    assert result.normalized_action["params"] == {
        "oxygen_device": None,
        "FiO2": None,
        "PEEP_used": None,
        "PEEP_cmH2O": None,
    }


def test_raw_text_in_clinician_medical_treatment_order_is_rejected() -> None:
    action = _valid_oxygen_order({"oxygen_device": "NRB"})
    action["raw_text"] = "put on oxygen"

    result = _validator().validate_clinician_proposal(action, GlobalState())

    assert result.ok is False
    assert any("raw_text" in error for error in result.errors)


def test_deep_nested_raw_text_in_params_is_rejected() -> None:
    result = _validator().validate_clinician_proposal(
        _valid_oxygen_order(
            {
                "oxygen_device": {
                    "value": "NRB",
                    "source": {"raw_text": "non-runtime text"},
                }
            }
        ),
        GlobalState(),
    )

    assert result.ok is False
    assert any("raw_text" in error for error in result.errors)


def test_raw_text_inside_verbal_action_is_rejected() -> None:
    result = _validator().validate_clinician_proposal(
        {
            "verbal_action": {
                "recipient": "patient",
                "content": "We are giving oxygen now.",
                "raw_text": "legacy raw text",
            },
            "action": _valid_oxygen_order({"oxygen_device": "NRB"}),
        },
        GlobalState(),
    )

    assert result.ok is False
    assert any("raw_text" in error for error in result.errors)


def test_non_dict_non_none_proposal_returns_clear_error() -> None:
    result = _validator().validate_clinician_proposal("hello", GlobalState())

    assert result.ok is False
    assert result.errors == ["Clinician proposal must be an object or null."]


def test_unknown_family_is_rejected() -> None:
    action = _valid_oxygen_order()
    action["family"] = "unknown_family"

    result = _validator().validate_medical_treatment_order(action)

    assert result.ok is False
    assert "Unknown action family 'unknown_family'." in result.errors


def test_unknown_kind_hint_is_rejected() -> None:
    action = _valid_oxygen_order()
    action["kind_hint"] = "unknown_kind"

    result = _validator().validate_medical_treatment_order(action)

    assert result.ok is False
    assert "Unknown kind_hint 'unknown_kind'." in result.errors


def test_family_kind_hint_mismatch_is_rejected() -> None:
    action = _valid_oxygen_order()
    action["family"] = ActionFamily.CARDIAC_RHYTHM

    result = _validator().validate_medical_treatment_order(action)

    assert result.ok is False
    assert any("family/kind_hint mismatch" in error for error in result.errors)


def test_no_action_proposed_by_clinician_as_treatment_is_rejected() -> None:
    result = _validator().validate_medical_treatment_order(
        {
            "type": "medical_treatment_order",
            "family": ActionFamily.TIME_PROGRESSION,
            "kind_hint": KindHint.NO_ACTION,
            "params": {"elapsed_min": 1},
        }
    )

    assert result.ok is False
    assert any("not clinician-selectable" in error for error in result.errors)


def test_extra_unknown_params_are_rejected() -> None:
    result = _validator().validate_medical_treatment_order(
        _valid_oxygen_order({"oxygen_device": "NRB", "unexpected": True})
    )

    assert result.ok is False
    assert "Unknown params for 'oxygen_support': ['unexpected']." in result.errors


def test_param_values_outside_param_options_are_not_rejected() -> None:
    result = _validator().validate_medical_treatment_order(
        _valid_oxygen_order({"oxygen_device": "custom_device", "FiO2": 0.37})
    )

    assert result.ok is True
    assert result.errors == []
    assert result.normalized_action["params"]["oxygen_device"] == "custom_device"
    assert result.normalized_action["params"]["FiO2"] == 0.37


def test_valid_diagnostic_order_passes_when_test_name_exists() -> None:
    result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "ECG"},
        _state_with_tests(),
    )

    assert result.ok is True
    assert result.action_type == "diagnostic_order"
    assert result.normalized_action == {
        "type": "diagnostic_order",
        "test_name": "ECG",
    }


def test_canonical_action_wrapper_path_works() -> None:
    result = _validator().validate_clinician_proposal(
        {"action": _valid_oxygen_order({"oxygen_device": "NRB"})},
        GlobalState(),
    )

    assert result.ok is True
    assert result.action_type == "medical_treatment_order"
    assert result.normalized_action["kind_hint"] == KindHint.OXYGEN_SUPPORT


def test_action_list_form_proposal_is_rejected() -> None:
    result = _validator().validate_clinician_proposal(
        {"action": [_valid_oxygen_order({"oxygen_device": "NRB"})]},
        GlobalState(),
    )

    assert result.ok is False
    assert any(
        "list-form action proposals are not supported" in error
        for error in result.errors
    )


def test_diagnostic_order_does_not_reveal_result_in_normalized_action() -> None:
    result = _validator().validate_diagnostic_order(
        {"type": "diagnostic_order", "test_name": "Chest X-ray"},
        _state_with_tests(),
    )

    assert result.ok is True
    assert "result" not in result.normalized_action
    assert result.normalized_action == {
        "type": "diagnostic_order",
        "test_name": "Chest X-ray",
    }


def test_direct_diagnostic_order_with_raw_text_is_rejected() -> None:
    result = _validator().validate_diagnostic_order(
        {"type": "diagnostic_order", "test_name": "ECG", "raw_text": "get ECG"},
        _state_with_tests(),
    )

    assert result.ok is False
    assert any("raw_text" in error for error in result.errors)


@pytest.mark.parametrize("test_name", [None, ""])
def test_diagnostic_order_with_missing_or_empty_test_name_is_rejected(
    test_name: str | None,
) -> None:
    action = {"type": "diagnostic_order"}
    if test_name is not None:
        action["test_name"] = test_name

    result = _validator().validate_diagnostic_order(action, _state_with_tests())

    assert result.ok is False
    assert "diagnostic_order.test_name is required." in result.errors


def test_unknown_diagnostic_test_name_is_rejected() -> None:
    result = _validator().validate_diagnostic_order(
        {"type": "diagnostic_order", "test_name": "Troponin"},
        _state_with_tests(),
    )

    assert result.ok is False
    assert "Unknown diagnostic test_name 'Troponin'." in result.errors


def test_proposal_with_both_treatment_and_diagnostic_order_is_rejected() -> None:
    result = _validator().validate_clinician_proposal(
        {
            "medical_treatment_order": _valid_oxygen_order(),
            "diagnostic_order": {"type": "diagnostic_order", "test_name": "ECG"},
        },
        _state_with_tests(),
    )

    assert result.ok is False
    assert any("at most one action" in error for error in result.errors)


def test_proposal_with_verbal_action_plus_one_action_is_valid() -> None:
    result = _validator().validate_clinician_proposal(
        {
            "verbal_action": {
                "recipient": "patient",
                "content": "We are giving oxygen now.",
            },
            "action": _valid_oxygen_order({"oxygen_device": "NRB"}),
        },
        GlobalState(),
    )

    assert result.ok is True
    assert result.action_type == "medical_treatment_order"


def test_validator_does_not_mutate_global_state() -> None:
    state = _state_with_tests()
    before = deepcopy(state.model_dump())

    result = _validator().validate_clinician_proposal(
        {"type": "diagnostic_order", "test_name": "ECG"},
        state,
    )

    assert result.ok is True
    assert state.model_dump() == before
    assert state.runtime_state.pending_diagnostic_results == []
    assert state.known_facts.available_results == []


def test_validator_does_not_import_transition_engines() -> None:
    source = inspect.getsource(validator_module)

    assert "transition_engines" not in source


def test_validator_only_creates_raw_text_none_for_normalized_treatment_action() -> None:
    medical_result = _validator().validate_medical_treatment_order(
        _valid_oxygen_order({"oxygen_device": "NRB"})
    )
    diagnostic_result = _validator().validate_diagnostic_order(
        {"type": "diagnostic_order", "test_name": "ECG"},
        _state_with_tests(),
    )

    assert medical_result.ok is True
    assert medical_result.normalized_action["raw_text"] is None
    assert "raw_text" not in medical_result.normalized_action["params"]
    assert diagnostic_result.ok is True
    assert "raw_text" not in diagnostic_result.normalized_action


def test_airway_management_validation_reflects_current_m2a_schema() -> None:
    valid_result = _validator().validate_medical_treatment_order(
        {
            "type": "medical_treatment_order",
            "family": ActionFamily.RESPIRATORY_SUPPORT,
            "kind_hint": KindHint.AIRWAY_MANAGEMENT,
            "params": {"stage": "preparation"},
        }
    )
    invalid_result = _validator().validate_medical_treatment_order(
        {
            "type": "medical_treatment_order",
            "family": ActionFamily.RESPIRATORY_SUPPORT,
            "kind_hint": KindHint.AIRWAY_MANAGEMENT,
            "params": {"procedure": "intubation"},
        }
    )

    assert valid_result.ok is True
    assert valid_result.normalized_action["params"] == {
        "stage": "preparation",
        "rsi_medication_used": None,
        "bvm_before": None,
        "intubated": None,
    }
    assert "procedure" not in valid_result.normalized_action["params"]
    assert invalid_result.ok is False
    assert "Unknown params for 'airway_management': ['procedure']." in (
        invalid_result.errors
    )


def test_oxygen_support_accepts_device_without_fixed_fio2_options() -> None:
    result = _validator().validate_medical_treatment_order(
        _valid_oxygen_order({"oxygen_device": "NRB", "FiO2": 0.83})
    )

    assert result.ok is True
    assert result.normalized_action["params"]["oxygen_device"] == "NRB"
    assert result.normalized_action["params"]["FiO2"] == 0.83


def test_fluid_bolus_does_not_require_volume_ml_from_fixed_options() -> None:
    result = _validator().validate_medical_treatment_order(
        {
            "type": "medical_treatment_order",
            "family": ActionFamily.CIRCULATION_HEMODYNAMICS,
            "kind_hint": KindHint.FLUID_BOLUS,
            "params": {"fluid_type": "normal_saline", "volume_ml": 1234},
        }
    )

    assert result.ok is True
    assert result.normalized_action["params"] == {
        "fluid_type": "normal_saline",
        "volume_ml": 1234,
    }


def test_diagnostic_turnaround_exists_but_validator_does_not_use_it() -> None:
    state = GlobalState(
        truth_state={
            "test_bank": [
                DiagnosticTestBankItem(
                    name="VBG",
                    result="pH 7.30",
                    turnaround_turns=2,
                )
            ]
        }
    )

    result = _validator().validate_diagnostic_order(
        {"type": "diagnostic_order", "test_name": "VBG"},
        state,
    )

    assert result.ok is True
    assert result.normalized_action == {
        "type": "diagnostic_order",
        "test_name": "VBG",
    }
    assert "turnaround_turns" not in result.normalized_action
    assert state.runtime_state.pending_diagnostic_results == []


def test_validation_result_is_frozen() -> None:
    result = ValidationResult(
        ok=True,
        action_type=None,
        normalized_action=None,
        errors=[],
    )

    with pytest.raises(FrozenInstanceError):
        result.ok = False
