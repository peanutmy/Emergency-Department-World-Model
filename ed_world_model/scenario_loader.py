"""Load extracted scenario JSON into a v1.3.1 GlobalState."""
from __future__ import annotations

from copy import deepcopy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ed_world_model.constants import DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS
from ed_world_model.state.global_state import (
    Demographics,
    Features,
    GlobalState,
    PatientInternalState,
    PatientState,
    RuntimeState,
    TestBankItem,
    TruthState,
    Vitals,
)

DEFAULT_DISCLOSURE_RULES = (
    "Use these hidden patient facts as free-form prompt guidance. Volunteer the "
    "chief complaint when appropriate and answer direct questions truthfully "
    "without exposing undisclosed facts to the clinical team."
)


class ScenarioLoader:
    """Convert existing extracted scenario JSON into initial runtime state.

    Transition pairs are evaluation data and are intentionally ignored by this
    loader. Runtime baseline physiology comes only from `case_context`
    scenario-level baseline fields.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        default_disclosure_rules: str = DEFAULT_DISCLOSURE_RULES,
    ) -> None:
        self.strict = strict
        self.default_disclosure_rules = default_disclosure_rules

    def load(
        self,
        source: str | Path | Mapping[str, Any],
        *,
        max_turns: int | None = None,
    ) -> GlobalState:
        if isinstance(source, Mapping):
            return self.load_dict(source, max_turns=max_turns)
        return self.load_file(source, max_turns=max_turns)

    def load_file(
        self,
        path: str | Path,
        *,
        max_turns: int | None = None,
    ) -> GlobalState:
        with Path(path).open("r", encoding="utf-8") as file:
            scenario = json.load(file)
        return self.load_dict(scenario, max_turns=max_turns)

    def load_dict(
        self,
        scenario: Mapping[str, Any],
        *,
        max_turns: int | None = None,
    ) -> GlobalState:
        scenario_data = deepcopy(dict(scenario))
        case_context = self._mapping_or_empty(
            scenario_data.get("case_context"),
            "case_context",
        )

        truth_state = TruthState(
            scenario_description=scenario_data.get("scenario_description"),
            demographics=self._build_demographics(case_context),
            patient_internal_state=self._build_patient_internal_state(case_context),
            test_bank=self._build_test_bank(case_context),
        )
        patient_state = PatientState(
            vitals=self._build_vitals(case_context),
            features=self._build_features(case_context),
        )

        runtime_state_data: dict[str, Any] = {"turn_index": 0}
        if max_turns is not None:
            runtime_state_data["max_turns"] = max_turns

        return GlobalState(
            truth_state=truth_state,
            patient_state=patient_state,
            runtime_state=RuntimeState(**runtime_state_data),
        )

    def _build_demographics(self, case_context: Mapping[str, Any]) -> Demographics:
        demographics = self._mapping_or_empty(
            case_context.get("demographics"),
            "case_context.demographics",
        )
        filtered = self._filter_model_fields(demographics, Demographics)
        return Demographics.model_validate(filtered)

    def _build_patient_internal_state(
        self,
        case_context: Mapping[str, Any],
    ) -> PatientInternalState:
        disclosure_rules = case_context.get("disclosure_rules")
        if disclosure_rules is None:
            disclosure_rules = self.default_disclosure_rules

        return PatientInternalState(
            hidden_history=self._string_list(
                case_context.get("history"),
                "case_context.history",
            ),
            hidden_allergies=self._string_list(
                case_context.get("allergies"),
                "case_context.allergies",
            ),
            hidden_home_medications=self._string_list(
                case_context.get("home_medications"),
                "case_context.home_medications",
            ),
            chief_complaint=case_context.get("chief_complaint"),
            symptoms=self._string_list(
                case_context.get("symptoms"),
                "case_context.symptoms",
            ),
            disclosure_rules=disclosure_rules,
        )

    def _build_test_bank(self, case_context: Mapping[str, Any]) -> list[TestBankItem]:
        supporting_findings = case_context.get("supporting_findings", [])
        if supporting_findings is None:
            return []
        if not isinstance(supporting_findings, list):
            if self.strict:
                raise ValueError("case_context.supporting_findings must be a list.")
            return []

        test_bank: list[TestBankItem] = []
        for finding in supporting_findings:
            if not isinstance(finding, Mapping):
                if self.strict:
                    raise ValueError(
                        "case_context.supporting_findings entries must be objects."
                    )
                continue

            name = finding.get("test_name") or finding.get("test_code")
            if not name:
                continue

            turnaround_turns = finding.get(
                "turnaround_turns",
                DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS,
            )
            if turnaround_turns is None:
                turnaround_turns = DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS

            test_bank.append(
                TestBankItem(
                    name=name,
                    result=finding.get("result_summary") or "",
                    turnaround_turns=turnaround_turns,
                )
            )

        return test_bank

    def _build_vitals(self, case_context: Mapping[str, Any]) -> Vitals:
        baseline_vitals = case_context.get("baseline_vitals")
        if baseline_vitals is None:
            if self.strict:
                raise ValueError("case_context.baseline_vitals is required.")
            return Vitals()
        baseline_mapping = self._mapping_or_empty(
            baseline_vitals,
            "case_context.baseline_vitals",
        )
        filtered = self._filter_model_fields(baseline_mapping, Vitals)
        return Vitals.model_validate(filtered)

    def _build_features(self, case_context: Mapping[str, Any]) -> Features:
        baseline_features = case_context.get("baseline_features")
        if baseline_features is None:
            if self.strict:
                raise ValueError("case_context.baseline_features is required.")
            return Features()
        baseline_mapping = self._mapping_or_empty(
            baseline_features,
            "case_context.baseline_features",
        )
        return Features.model_validate(dict(baseline_mapping))

    def _mapping_or_empty(self, value: Any, field_name: str) -> Mapping[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return value
        if self.strict:
            raise ValueError(f"{field_name} must be an object.")
        return {}

    def _string_list(self, value: Any, field_name: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if self.strict:
            raise ValueError(f"{field_name} must be a list.")
        return []

    def _filter_model_fields(
        self,
        data: Mapping[str, Any],
        model_type: type[Demographics] | type[Vitals],
    ) -> dict[str, Any]:
        return {key: data[key] for key in model_type.model_fields if key in data}


def load_scenario(
    source: str | Path | Mapping[str, Any],
    *,
    max_turns: int | None = None,
    strict: bool = False,
) -> GlobalState:
    return ScenarioLoader(strict=strict).load(source, max_turns=max_turns)


__all__ = [
    "DEFAULT_DISCLOSURE_RULES",
    "ScenarioLoader",
    "load_scenario",
]
