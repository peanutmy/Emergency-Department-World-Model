from __future__ import annotations

from collections import Counter
from pathlib import Path
import inspect
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.actions.registry as registry_module
from ed_world_model.actions.registry import (
    ActionDefinition,
    ActionRegistry,
    KindHint,
)


EXPECTED_KIND_HINTS = {
    "no_action",
    "fluid_bolus",
    "airway_management",
    "oxygen_support",
    "vasopressor",
    "airway_medication",
    "vasodilator",
    "blood_transfusion",
    "antidote",
    "membrane_stabilization",
    "bronchodilator",
    "rate_control",
    "tube_thoracostomy",
    "surgical_airway",
    "sedation",
    "anaphylaxis_treatment",
    "uterine_exploration",
    "transcutaneous_pacing",
    "synchronized_cardioversion",
    "prostaglandin_infusion",
    "osmotherapy",
    "oral_airway",
    "obstetric_delivery",
    "needle_decompression",
    "inotrope",
    "hyperkalemia_shift",
    "electrolyte_correction",
    "corticosteroid",
    "chronotropic_agent",
    "anticonvulsant",
    "anticoagulation",
    "antiarrhythmic",
    "analgesia",
}

MEDICATION_LIKE_KIND_HINTS = {
    "vasopressor",
    "airway_medication",
    "vasodilator",
    "antidote",
    "membrane_stabilization",
    "bronchodilator",
    "rate_control",
    "sedation",
    "anaphylaxis_treatment",
    "prostaglandin_infusion",
    "osmotherapy",
    "inotrope",
    "hyperkalemia_shift",
    "electrolyte_correction",
    "corticosteroid",
    "chronotropic_agent",
    "anticonvulsant",
    "anticoagulation",
    "antiarrhythmic",
    "analgesia",
}

PROCEDURE_LIKE_KIND_HINTS = {
    "tube_thoracostomy",
    "surgical_airway",
    "uterine_exploration",
    "transcutaneous_pacing",
    "synchronized_cardioversion",
    "oral_airway",
    "obstetric_delivery",
    "needle_decompression",
}


def test_registry_includes_exactly_33_kind_hint_values() -> None:
    registry = ActionRegistry()

    assert len(registry.list_kind_hints()) == 33


def test_all_expected_kind_hint_values_are_present() -> None:
    registry = ActionRegistry()

    assert set(registry.list_kind_hints()) == EXPECTED_KIND_HINTS


def test_list_families_returns_non_empty_families() -> None:
    registry = ActionRegistry()

    assert registry.list_families()
    assert all(family for family in registry.list_families())


def test_every_kind_hint_belongs_to_exactly_one_family() -> None:
    registry = ActionRegistry()
    assigned_kind_hints = [
        kind_hint
        for family in registry.list_families()
        for kind_hint in registry.list_kind_hints_for_family(family)
    ]

    counts = Counter(assigned_kind_hints)

    assert set(counts) == EXPECTED_KIND_HINTS
    assert all(count == 1 for count in counts.values())


def test_each_family_maps_to_at_least_one_kind_hint() -> None:
    registry = ActionRegistry()

    for family in registry.list_families():
        assert registry.list_kind_hints_for_family(family)


def test_oxygen_support_params_template_has_expected_keys() -> None:
    registry = ActionRegistry()

    assert set(registry.get_params_template(KindHint.OXYGEN_SUPPORT)) == {
        "oxygen_device",
        "FiO2",
        "PEEP_used",
        "PEEP_cmH2O",
    }


def test_airway_management_params_template_has_expected_keys() -> None:
    registry = ActionRegistry()

    assert set(registry.get_params_template(KindHint.AIRWAY_MANAGEMENT)) == {
        "stage",
        "rsi_medication_used",
        "bvm_before",
        "intubated",
    }


def test_medication_like_kind_hints_use_medication_params_template() -> None:
    registry = ActionRegistry()

    for kind_hint in MEDICATION_LIKE_KIND_HINTS:
        assert registry.get_params_template(kind_hint) == {
            "drug_name": None,
            "dose": None,
            "unit": None,
        }


def test_procedure_like_kind_hints_use_procedure_params_template() -> None:
    registry = ActionRegistry()

    for kind_hint in PROCEDURE_LIKE_KIND_HINTS:
        assert registry.get_params_template(kind_hint) == {
            "site": None,
            "side": None,
            "device": None,
            "energy_J": None,
        }


def test_no_action_params_template_has_elapsed_min_one() -> None:
    registry = ActionRegistry()

    assert registry.get_params_template(KindHint.NO_ACTION) == {"elapsed_min": 1}


def test_fluid_bolus_has_fluid_type_options_but_no_volume_options() -> None:
    registry = ActionRegistry()

    assert registry.get_param_options(KindHint.FLUID_BOLUS) == {
        "fluid_type": ["normal_saline", "lactated_ringers", None],
    }
    assert "volume_ml" not in registry.get_param_options(KindHint.FLUID_BOLUS)
    assert "Open numeric field" in registry.get_param_guidance(
        KindHint.FLUID_BOLUS
    )["volume_ml"]


def test_oxygen_support_has_device_options_but_no_fixed_fio2_or_peep_options() -> None:
    registry = ActionRegistry()
    options = registry.get_param_options(KindHint.OXYGEN_SUPPORT)
    guidance = registry.get_param_guidance(KindHint.OXYGEN_SUPPORT)

    assert options == {
        "oxygen_device": [
            "nasal_cannula",
            "simple_mask",
            "NRB",
            "BVM",
            "BiPAP",
            "vent",
            None,
        ],
    }
    assert "FiO2" not in options
    assert "PEEP_used" not in options
    assert "PEEP_cmH2O" not in options
    assert "device-dependent" in guidance["FiO2"].lower()
    assert "Open boolean/null" in guidance["PEEP_used"]
    assert "Open numeric/null" in guidance["PEEP_cmH2O"]


def test_airway_management_stage_options_are_canonical() -> None:
    registry = ActionRegistry()
    options = registry.get_param_options(KindHint.AIRWAY_MANAGEMENT)

    assert options["stage"] == [
        "preparation",
        "attempt",
        "completed",
        None,
    ]
    assert options["rsi_medication_used"] == [True, False, None]
    assert options["bvm_before"] == [True, False, None]
    assert options["intubated"] == [True, False, None]
    assert "procedure" not in registry.get_params_template(
        KindHint.AIRWAY_MANAGEMENT
    )
    assert "preparation = preparing but not intubated" in registry.get_param_guidance(
        KindHint.AIRWAY_MANAGEMENT
    )["stage"]


def test_blood_transfusion_has_blood_product_options_only() -> None:
    registry = ActionRegistry()

    assert registry.get_param_options(KindHint.BLOOD_TRANSFUSION) == {
        "blood_product": ["packed_rbc", "whole_blood", "plasma", "platelets", None],
    }
    assert "units" not in registry.get_param_options(KindHint.BLOOD_TRANSFUSION)
    assert "strategy" not in registry.get_param_options(KindHint.BLOOD_TRANSFUSION)
    assert set(registry.get_param_guidance(KindHint.BLOOD_TRANSFUSION)) == {
        "units",
        "strategy",
    }


def test_procedure_like_kind_hints_have_side_options_only() -> None:
    registry = ActionRegistry()

    for kind_hint in PROCEDURE_LIKE_KIND_HINTS:
        assert registry.get_param_options(kind_hint) == {
            "side": ["left", "right", "bilateral", None],
        }
        guidance = registry.get_param_guidance(kind_hint)
        assert set(guidance) == {"site", "device", "energy_J"}
        assert "site" not in registry.get_param_options(kind_hint)
        assert "device" not in registry.get_param_options(kind_hint)
        assert "energy_J" not in registry.get_param_options(kind_hint)


def test_medication_like_kind_hints_have_no_fixed_drug_dose_or_unit_options() -> None:
    registry = ActionRegistry()

    for kind_hint in MEDICATION_LIKE_KIND_HINTS:
        assert registry.get_param_options(kind_hint) == {}
        assert set(registry.get_param_guidance(kind_hint)) == {
            "drug_name",
            "dose",
            "unit",
        }


def test_no_action_is_not_clinician_selectable() -> None:
    registry = ActionRegistry()

    definition = registry.get_definition(KindHint.NO_ACTION)

    assert definition.clinician_selectable is False
    assert "System-generated" in registry.get_param_guidance(KindHint.NO_ACTION)[
        "elapsed_min"
    ]


def test_param_options_are_documented_as_guidance_not_validation_constraints() -> None:
    registry = ActionRegistry()

    assert "menu guidance only" in inspect.getdoc(ActionDefinition)
    assert "not validation constraints" in inspect.getdoc(ActionDefinition)
    assert not hasattr(registry, "validate")


def test_get_definition_returns_definition_for_known_kind_hint() -> None:
    registry = ActionRegistry()

    definition = registry.get_definition(KindHint.OXYGEN_SUPPORT)

    assert isinstance(definition, ActionDefinition)
    assert definition.kind_hint == KindHint.OXYGEN_SUPPORT
    assert definition.family == "respiratory_support"


def test_get_definition_raises_clear_error_for_unknown_kind_hint() -> None:
    registry = ActionRegistry()

    with pytest.raises(ValueError, match="Unknown kind_hint 'made_up'"):
        registry.get_definition("made_up")


def test_runtime_raw_text_is_not_part_of_definitions_or_params_templates() -> None:
    registry = ActionRegistry()

    assert "raw_text" not in ActionDefinition.__dataclass_fields__
    for kind_hint in registry.list_kind_hints():
        definition = registry.get_definition(kind_hint)
        assert "raw_text" not in definition.params_template
        assert "raw_text" not in definition.param_options
        assert "raw_text" not in definition.param_guidance
        assert "raw_text" not in registry.get_params_template(kind_hint)
        assert "raw_text" not in registry.get_param_options(kind_hint)
        assert "raw_text" not in registry.get_param_guidance(kind_hint)


def test_registry_does_not_import_or_mutate_global_state() -> None:
    source = inspect.getsource(registry_module)

    assert "GlobalState" not in source
    assert "ed_world_model.state" not in source


def test_registry_does_not_import_transition_engines() -> None:
    source = inspect.getsource(registry_module)

    assert "transition_engines" not in source
