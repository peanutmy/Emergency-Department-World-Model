"""Validation for clinician action proposals in the v1.3.1 runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ed_world_model.actions.registry import ActionRegistry


@dataclass(frozen=True)
class ValidationResult:
    """Lightweight validation outcome for action proposals."""

    ok: bool
    action_type: str | None
    normalized_action: dict[str, Any] | None
    errors: list[str]


class ActionValidator:
    """Validates clinician proposals without executing actions or mutating state."""

    def __init__(self, registry: ActionRegistry) -> None:
        self._registry = registry

    def validate_clinician_proposal(
        self,
        proposal: Any,
        global_state: Any,
    ) -> ValidationResult:
        if _contains_key(proposal, "raw_text"):
            return _invalid(
                "Clinician proposals must not include raw_text; runtime raw_text "
                "is system-controlled and always null."
            )

        action, errors = self._extract_clinician_action(proposal)
        if errors:
            return _invalid(*errors)
        if _is_null_action(action):
            return ValidationResult(
                ok=True,
                action_type=None,
                normalized_action=None,
                errors=[],
            )
        if not isinstance(action, dict):
            return _invalid("Clinician action must be an object or null.")

        action_type = action.get("type")
        if action_type == "medical_treatment_order":
            return self.validate_medical_treatment_order(action)
        if action_type == "diagnostic_order":
            return self.validate_diagnostic_order(action, global_state)
        return _invalid(
            "Unknown clinician action type "
            f"{action_type!r}; expected medical_treatment_order, "
            "diagnostic_order, or null."
        )

    def validate_medical_treatment_order(self, action: Any) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(action, dict):
            return _invalid("Medical treatment order must be an object.")
        if _contains_key(action, "raw_text"):
            errors.append(
                "Clinician medical_treatment_order must not include raw_text."
            )

        action_type = action.get("type")
        if action_type != "medical_treatment_order":
            errors.append(
                "Medical treatment order type must be "
                "'medical_treatment_order'."
            )

        family = action.get("family")
        kind_hint = action.get("kind_hint")
        if not isinstance(family, str) or not family:
            errors.append("medical_treatment_order.family is required.")
        elif not self._registry.is_known_family(family):
            errors.append(f"Unknown action family {family!r}.")

        definition = None
        if not isinstance(kind_hint, str) or not kind_hint:
            errors.append("medical_treatment_order.kind_hint is required.")
        elif not self._registry.is_known_kind_hint(kind_hint):
            errors.append(f"Unknown kind_hint {kind_hint!r}.")
        else:
            definition = self._registry.get_definition(kind_hint)

        if definition is not None:
            if isinstance(family, str) and self._registry.is_known_family(family):
                if definition.family != family:
                    errors.append(
                        "family/kind_hint mismatch: "
                        f"{kind_hint!r} belongs to {definition.family!r}, "
                        f"not {family!r}."
                    )
            if not definition.clinician_selectable:
                errors.append(
                    f"kind_hint {kind_hint!r} is system-generated and is not "
                    "clinician-selectable."
                )

        params = action.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            errors.append("medical_treatment_order.params must be an object if set.")
            params = {}

        normalized_params: dict[str, Any] = {}
        if definition is not None:
            template = self._registry.get_params_template(definition.kind_hint)
            extra_params = sorted(set(params) - set(template))
            if extra_params:
                errors.append(
                    f"Unknown params for {definition.kind_hint!r}: {extra_params}."
                )
            normalized_params = {
                param_name: params.get(param_name, template_value)
                for param_name, template_value in template.items()
            }

        if errors:
            return ValidationResult(
                ok=False,
                action_type="medical_treatment_order",
                normalized_action=None,
                errors=errors,
            )

        return ValidationResult(
            ok=True,
            action_type="medical_treatment_order",
            normalized_action={
                "raw_text": None,
                "kind_hint": definition.kind_hint,
                "params": normalized_params,
            },
            errors=[],
        )

    def validate_diagnostic_order(
        self,
        action: Any,
        global_state: Any,
    ) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(action, dict):
            return _invalid("Diagnostic order must be an object.")
        if _contains_key(action, "raw_text"):
            errors.append("Clinician diagnostic_order must not include raw_text.")

        action_type = action.get("type")
        if action_type != "diagnostic_order":
            errors.append("Diagnostic order type must be 'diagnostic_order'.")

        test_name = action.get("test_name")
        if not isinstance(test_name, str) or not test_name:
            errors.append("diagnostic_order.test_name is required.")
        else:
            test_names = self._test_bank_names(global_state)
            if test_names is None:
                errors.append("global_state.truth_state.test_bank is required.")
            elif test_name not in test_names:
                errors.append(f"Unknown diagnostic test_name {test_name!r}.")

        if errors:
            return ValidationResult(
                ok=False,
                action_type="diagnostic_order",
                normalized_action=None,
                errors=errors,
            )

        return ValidationResult(
            ok=True,
            action_type="diagnostic_order",
            normalized_action={
                "type": "diagnostic_order",
                "test_name": test_name,
            },
            errors=[],
        )

    def _extract_clinician_action(self, proposal: Any) -> tuple[Any, list[str]]:
        if proposal is None:
            return None, []
        if not isinstance(proposal, dict):
            return None, ["Clinician proposal must be an object or null."]

        if "type" in proposal and "action" not in proposal:
            return proposal, []

        actions: list[Any] = []
        null_action_seen = False

        if "action" in proposal:
            action_value = proposal["action"]
            if isinstance(action_value, list):
                return (
                    None,
                    [
                        "proposal.action must be a single action object or null; "
                        "list-form action proposals are not supported."
                    ],
                )
            if _is_null_action(action_value):
                null_action_seen = True
            else:
                actions.append(action_value)

        for action_key in ("medical_treatment_order", "diagnostic_order"):
            if action_key not in proposal:
                continue
            action_value = proposal[action_key]
            if _is_null_action(action_value):
                null_action_seen = True
                continue
            actions.append(action_value)

        if len(actions) > 1:
            return (
                None,
                [
                    "Clinician may produce at most one action per turn; "
                    "proposal contains multiple actions."
                ],
            )
        if actions:
            return actions[0], []
        if null_action_seen:
            return None, []
        return None, []

    @staticmethod
    def _test_bank_names(global_state: Any) -> set[str] | None:
        truth_state = getattr(global_state, "truth_state", None)
        test_bank = getattr(truth_state, "test_bank", None)
        if test_bank is None:
            return None

        test_names: set[str] = set()
        for test in test_bank:
            name = getattr(test, "name", None)
            if isinstance(test, dict):
                name = test.get("name")
            if isinstance(name, str):
                test_names.add(name)
        return test_names


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _is_null_action(action: Any) -> bool:
    return action is None or (
        isinstance(action, dict) and "type" in action and action.get("type") is None
    )


def _invalid(*errors: str) -> ValidationResult:
    return ValidationResult(
        ok=False,
        action_type=None,
        normalized_action=None,
        errors=list(errors),
    )


__all__ = ["ActionValidator", "ValidationResult"]
