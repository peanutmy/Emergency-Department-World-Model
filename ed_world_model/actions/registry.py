"""Action taxonomy and params templates for v1.3.1 runtime actions."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


EMPTY_MAPPING = MappingProxyType({})


class ActionFamily:
    TIME_PROGRESSION = "time_progression"
    RESPIRATORY_SUPPORT = "respiratory_support"
    CIRCULATION_HEMODYNAMICS = "circulation_hemodynamics"
    CARDIAC_RHYTHM = "cardiac_rhythm"
    MEDICATION_TOXICOLOGY_METABOLIC = "medication_toxicology_metabolic"
    PROCEDURES = "procedures"


class KindHint:
    NO_ACTION = "no_action"
    FLUID_BOLUS = "fluid_bolus"
    AIRWAY_MANAGEMENT = "airway_management"
    OXYGEN_SUPPORT = "oxygen_support"
    VASOPRESSOR = "vasopressor"
    AIRWAY_MEDICATION = "airway_medication"
    VASODILATOR = "vasodilator"
    BLOOD_TRANSFUSION = "blood_transfusion"
    ANTIDOTE = "antidote"
    MEMBRANE_STABILIZATION = "membrane_stabilization"
    BRONCHODILATOR = "bronchodilator"
    RATE_CONTROL = "rate_control"
    TUBE_THORACOSTOMY = "tube_thoracostomy"
    SURGICAL_AIRWAY = "surgical_airway"
    SEDATION = "sedation"
    ANAPHYLAXIS_TREATMENT = "anaphylaxis_treatment"
    UTERINE_EXPLORATION = "uterine_exploration"
    TRANSCUTANEOUS_PACING = "transcutaneous_pacing"
    SYNCHRONIZED_CARDIOVERSION = "synchronized_cardioversion"
    PROSTAGLANDIN_INFUSION = "prostaglandin_infusion"
    OSMOTHERAPY = "osmotherapy"
    ORAL_AIRWAY = "oral_airway"
    OBSTETRIC_DELIVERY = "obstetric_delivery"
    NEEDLE_DECOMPRESSION = "needle_decompression"
    INOTROPE = "inotrope"
    HYPERKALEMIA_SHIFT = "hyperkalemia_shift"
    ELECTROLYTE_CORRECTION = "electrolyte_correction"
    CORTICOSTEROID = "corticosteroid"
    CHRONOTROPIC_AGENT = "chronotropic_agent"
    ANTICONVULSANT = "anticonvulsant"
    ANTICOAGULATION = "anticoagulation"
    ANTIARRHYTHMIC = "antiarrhythmic"
    ANALGESIA = "analgesia"


@dataclass(frozen=True)
class ActionDefinition:
    """Registry entry for prompt guidance and runtime action field shape.

    `param_options` are menu guidance only. They are not validation constraints
    in v1.3.1; proposal validation belongs to ActionValidator.
    """

    kind_hint: str
    family: str
    params_template: Mapping[str, Any]
    param_options: Mapping[str, tuple[Any, ...]] = EMPTY_MAPPING
    param_guidance: Mapping[str, str] = EMPTY_MAPPING
    clinician_selectable: bool = True
    description: str | None = None


OXYGEN_SUPPORT_PARAMS_TEMPLATE = MappingProxyType(
    {
        "oxygen_device": None,
        "FiO2": None,
        "PEEP_used": None,
        "PEEP_cmH2O": None,
    }
)

AIRWAY_MANAGEMENT_PARAMS_TEMPLATE = MappingProxyType(
    {
        "stage": None,
        "rsi_medication_used": None,
        "bvm_before": None,
        "intubated": None,
    }
)

FLUID_BOLUS_PARAMS_TEMPLATE = MappingProxyType(
    {
        "fluid_type": None,
        "volume_ml": None,
    }
)

BLOOD_TRANSFUSION_PARAMS_TEMPLATE = MappingProxyType(
    {
        "blood_product": None,
        "units": None,
        "strategy": None,
    }
)

MEDICATION_LIKE_PARAMS_TEMPLATE = MappingProxyType(
    {
        "drug_name": None,
        "dose": None,
        "unit": None,
    }
)

PROCEDURE_LIKE_PARAMS_TEMPLATE = MappingProxyType(
    {
        "site": None,
        "side": None,
        "device": None,
        "energy_J": None,
    }
)

NO_ACTION_PARAMS_TEMPLATE = MappingProxyType({"elapsed_min": 1})

OXYGEN_SUPPORT_PARAM_OPTIONS = MappingProxyType(
    {
        "oxygen_device": (
            "nasal_cannula",
            "simple_mask",
            "NRB",
            "BVM",
            "BiPAP",
            "vent",
            None,
        ),
    }
)

OXYGEN_SUPPORT_PARAM_GUIDANCE = MappingProxyType(
    {
        "FiO2": (
            "Device-dependent approximate value; may be null if unspecified."
        ),
        "PEEP_used": (
            "Open boolean/null; true when PEEP is explicitly part of the "
            "intervention."
        ),
        "PEEP_cmH2O": "Open numeric/null; null if unspecified.",
    }
)

AIRWAY_MANAGEMENT_PARAM_OPTIONS = MappingProxyType(
    {
        "stage": ("preparation", "attempt", "completed", None),
        "rsi_medication_used": (True, False, None),
        "bvm_before": (True, False, None),
        "intubated": (True, False, None),
    }
)

AIRWAY_MANAGEMENT_PARAM_GUIDANCE = MappingProxyType(
    {
        "stage": (
            "preparation = preparing but not intubated; attempt = "
            "attempting/ongoing, completion unclear; completed = intubation "
            "completed."
        ),
    }
)

FLUID_BOLUS_PARAM_OPTIONS = MappingProxyType(
    {
        "fluid_type": ("normal_saline", "lactated_ringers", None),
    }
)

FLUID_BOLUS_PARAM_GUIDANCE = MappingProxyType(
    {
        "volume_ml": (
            "Open numeric field. Clinician should choose based on patient "
            "condition; may be null if unspecified."
        ),
    }
)

BLOOD_TRANSFUSION_PARAM_OPTIONS = MappingProxyType(
    {
        "blood_product": ("packed_rbc", "whole_blood", "plasma", "platelets", None),
    }
)

BLOOD_TRANSFUSION_PARAM_GUIDANCE = MappingProxyType(
    {
        "units": "Open numeric/null field; may be null if unspecified.",
        "strategy": "Open text/null field; may be null if unspecified.",
    }
)

MEDICATION_LIKE_PARAM_GUIDANCE = MappingProxyType(
    {
        "drug_name": "Open medication name field; may be null if unspecified.",
        "dose": "Open dose field; may be null if unspecified.",
        "unit": "Open unit field; may be null if unspecified.",
    }
)

PROCEDURE_LIKE_PARAM_OPTIONS = MappingProxyType(
    {
        "side": ("left", "right", "bilateral", None),
    }
)

PROCEDURE_LIKE_PARAM_GUIDANCE = MappingProxyType(
    {
        "site": "Open site field; may be null if unspecified.",
        "device": "Open device field; may be null if unspecified.",
        "energy_J": "Open numeric/null field; may be null if unspecified.",
    }
)

NO_ACTION_PARAM_GUIDANCE = MappingProxyType(
    {
        "elapsed_min": (
            "System-generated one-minute progression; not clinician-selectable."
        ),
    }
)

MEDICATION_LIKE_KIND_HINTS = (
    KindHint.VASOPRESSOR,
    KindHint.AIRWAY_MEDICATION,
    KindHint.VASODILATOR,
    KindHint.ANTIDOTE,
    KindHint.MEMBRANE_STABILIZATION,
    KindHint.BRONCHODILATOR,
    KindHint.RATE_CONTROL,
    KindHint.SEDATION,
    KindHint.ANAPHYLAXIS_TREATMENT,
    KindHint.PROSTAGLANDIN_INFUSION,
    KindHint.OSMOTHERAPY,
    KindHint.INOTROPE,
    KindHint.HYPERKALEMIA_SHIFT,
    KindHint.ELECTROLYTE_CORRECTION,
    KindHint.CORTICOSTEROID,
    KindHint.CHRONOTROPIC_AGENT,
    KindHint.ANTICONVULSANT,
    KindHint.ANTICOAGULATION,
    KindHint.ANTIARRHYTHMIC,
    KindHint.ANALGESIA,
)

PROCEDURE_LIKE_KIND_HINTS = (
    KindHint.TUBE_THORACOSTOMY,
    KindHint.SURGICAL_AIRWAY,
    KindHint.UTERINE_EXPLORATION,
    KindHint.TRANSCUTANEOUS_PACING,
    KindHint.SYNCHRONIZED_CARDIOVERSION,
    KindHint.ORAL_AIRWAY,
    KindHint.OBSTETRIC_DELIVERY,
    KindHint.NEEDLE_DECOMPRESSION,
)


def _definition(
    kind_hint: str,
    family: str,
    params_template: Mapping[str, Any],
    *,
    param_options: Mapping[str, tuple[Any, ...]] = EMPTY_MAPPING,
    param_guidance: Mapping[str, str] = EMPTY_MAPPING,
    clinician_selectable: bool = True,
    description: str | None = None,
) -> ActionDefinition:
    return ActionDefinition(
        kind_hint=kind_hint,
        family=family,
        params_template=params_template,
        param_options=param_options,
        param_guidance=param_guidance,
        clinician_selectable=clinician_selectable,
        description=description,
    )


DEFAULT_ACTION_DEFINITIONS = (
    _definition(
        KindHint.NO_ACTION,
        ActionFamily.TIME_PROGRESSION,
        NO_ACTION_PARAMS_TEMPLATE,
        param_guidance=NO_ACTION_PARAM_GUIDANCE,
        clinician_selectable=False,
        description="System-generated no-action progression.",
    ),
    _definition(
        KindHint.OXYGEN_SUPPORT,
        ActionFamily.RESPIRATORY_SUPPORT,
        OXYGEN_SUPPORT_PARAMS_TEMPLATE,
        param_options=OXYGEN_SUPPORT_PARAM_OPTIONS,
        param_guidance=OXYGEN_SUPPORT_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.AIRWAY_MANAGEMENT,
        ActionFamily.RESPIRATORY_SUPPORT,
        AIRWAY_MANAGEMENT_PARAMS_TEMPLATE,
        param_options=AIRWAY_MANAGEMENT_PARAM_OPTIONS,
        param_guidance=AIRWAY_MANAGEMENT_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.AIRWAY_MEDICATION,
        ActionFamily.RESPIRATORY_SUPPORT,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.BRONCHODILATOR,
        ActionFamily.RESPIRATORY_SUPPORT,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.SURGICAL_AIRWAY,
        ActionFamily.RESPIRATORY_SUPPORT,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ORAL_AIRWAY,
        ActionFamily.RESPIRATORY_SUPPORT,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.FLUID_BOLUS,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        FLUID_BOLUS_PARAMS_TEMPLATE,
        param_options=FLUID_BOLUS_PARAM_OPTIONS,
        param_guidance=FLUID_BOLUS_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.VASOPRESSOR,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.VASODILATOR,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.INOTROPE,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.CHRONOTROPIC_AGENT,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.BLOOD_TRANSFUSION,
        ActionFamily.CIRCULATION_HEMODYNAMICS,
        BLOOD_TRANSFUSION_PARAMS_TEMPLATE,
        param_options=BLOOD_TRANSFUSION_PARAM_OPTIONS,
        param_guidance=BLOOD_TRANSFUSION_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.RATE_CONTROL,
        ActionFamily.CARDIAC_RHYTHM,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANTIARRHYTHMIC,
        ActionFamily.CARDIAC_RHYTHM,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.SYNCHRONIZED_CARDIOVERSION,
        ActionFamily.CARDIAC_RHYTHM,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.TRANSCUTANEOUS_PACING,
        ActionFamily.CARDIAC_RHYTHM,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANTIDOTE,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.MEMBRANE_STABILIZATION,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.HYPERKALEMIA_SHIFT,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ELECTROLYTE_CORRECTION,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.CORTICOSTEROID,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANTICONVULSANT,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANTICOAGULATION,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANALGESIA,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.SEDATION,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.ANAPHYLAXIS_TREATMENT,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.PROSTAGLANDIN_INFUSION,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.OSMOTHERAPY,
        ActionFamily.MEDICATION_TOXICOLOGY_METABOLIC,
        MEDICATION_LIKE_PARAMS_TEMPLATE,
        param_guidance=MEDICATION_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.TUBE_THORACOSTOMY,
        ActionFamily.PROCEDURES,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.NEEDLE_DECOMPRESSION,
        ActionFamily.PROCEDURES,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.UTERINE_EXPLORATION,
        ActionFamily.PROCEDURES,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
    _definition(
        KindHint.OBSTETRIC_DELIVERY,
        ActionFamily.PROCEDURES,
        PROCEDURE_LIKE_PARAMS_TEMPLATE,
        param_options=PROCEDURE_LIKE_PARAM_OPTIONS,
        param_guidance=PROCEDURE_LIKE_PARAM_GUIDANCE,
    ),
)


class ActionRegistry:
    """Lightweight lookup registry for action families and kind hints."""

    def __init__(
        self,
        definitions: tuple[ActionDefinition, ...] = DEFAULT_ACTION_DEFINITIONS,
    ) -> None:
        definitions_by_kind_hint: dict[str, ActionDefinition] = {}
        family_to_kind_hints: dict[str, list[str]] = {}

        for definition in definitions:
            if definition.kind_hint in definitions_by_kind_hint:
                raise ValueError(f"Duplicate kind_hint {definition.kind_hint!r}.")
            definitions_by_kind_hint[definition.kind_hint] = definition
            family_to_kind_hints.setdefault(definition.family, []).append(
                definition.kind_hint
            )

        self._definitions_by_kind_hint = definitions_by_kind_hint
        self._family_to_kind_hints = {
            family: tuple(kind_hints)
            for family, kind_hints in family_to_kind_hints.items()
        }

    def list_families(self) -> list[str]:
        return list(self._family_to_kind_hints)

    def list_kind_hints(self) -> list[str]:
        return list(self._definitions_by_kind_hint)

    def list_kind_hints_for_family(self, family: str) -> list[str]:
        if family not in self._family_to_kind_hints:
            raise ValueError(f"Unknown action family {family!r}.")
        return list(self._family_to_kind_hints[family])

    def get_definition(self, kind_hint: str) -> ActionDefinition:
        try:
            return self._definitions_by_kind_hint[kind_hint]
        except KeyError as exc:
            raise ValueError(f"Unknown kind_hint {kind_hint!r}.") from exc

    def get_params_template(self, kind_hint: str) -> dict[str, Any]:
        return dict(self.get_definition(kind_hint).params_template)

    def get_param_options(self, kind_hint: str) -> dict[str, list[Any]]:
        definition = self.get_definition(kind_hint)
        return {
            param_name: list(options)
            for param_name, options in definition.param_options.items()
        }

    def get_param_guidance(self, kind_hint: str) -> dict[str, str]:
        return dict(self.get_definition(kind_hint).param_guidance)

    def is_known_kind_hint(self, kind_hint: str) -> bool:
        return kind_hint in self._definitions_by_kind_hint

    def is_known_family(self, family: str) -> bool:
        return family in self._family_to_kind_hints


__all__ = [
    "AIRWAY_MANAGEMENT_PARAMS_TEMPLATE",
    "ActionDefinition",
    "ActionFamily",
    "ActionRegistry",
    "AIRWAY_MANAGEMENT_PARAM_GUIDANCE",
    "AIRWAY_MANAGEMENT_PARAM_OPTIONS",
    "BLOOD_TRANSFUSION_PARAMS_TEMPLATE",
    "BLOOD_TRANSFUSION_PARAM_GUIDANCE",
    "BLOOD_TRANSFUSION_PARAM_OPTIONS",
    "DEFAULT_ACTION_DEFINITIONS",
    "EMPTY_MAPPING",
    "FLUID_BOLUS_PARAMS_TEMPLATE",
    "FLUID_BOLUS_PARAM_GUIDANCE",
    "FLUID_BOLUS_PARAM_OPTIONS",
    "KindHint",
    "MEDICATION_LIKE_KIND_HINTS",
    "MEDICATION_LIKE_PARAM_GUIDANCE",
    "MEDICATION_LIKE_PARAMS_TEMPLATE",
    "NO_ACTION_PARAMS_TEMPLATE",
    "NO_ACTION_PARAM_GUIDANCE",
    "OXYGEN_SUPPORT_PARAMS_TEMPLATE",
    "OXYGEN_SUPPORT_PARAM_GUIDANCE",
    "OXYGEN_SUPPORT_PARAM_OPTIONS",
    "PROCEDURE_LIKE_KIND_HINTS",
    "PROCEDURE_LIKE_PARAM_GUIDANCE",
    "PROCEDURE_LIKE_PARAM_OPTIONS",
    "PROCEDURE_LIKE_PARAMS_TEMPLATE",
]
