"""Deterministic rule-based engine for educational ED transition pairs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .common import CANONICAL_VITAL_KEYS, EngineOutput, VitalKey


VITAL_BOUNDS: dict[VitalKey, tuple[float, float]] = {
    "HR": (0, 250),
    "BP_sys": (40, 260),
    "BP_dia": (20, 160),
    "RR": (0, 60),
    "O2Sat": (0, 100),
    "T": (30, 43),
}

OXYGEN_DEVICE_DEFAULT_FIO2 = {
    "room_air": 0.21,
    "nasal_cannula": 0.28,
    "simple_mask": 0.4,
    "NRB": 0.8,
    "BVM": 1.0,
    "BiPAP": 1.0,
    "HFNC": 1.0,
    "vent": 1.0,
}

OXYGEN_DEVICE_ALIASES = {
    "room air": "room_air",
    "room_air": "room_air",
    "nasal cannula": "nasal_cannula",
    "nasal_cannula": "nasal_cannula",
    "nc": "nasal_cannula",
    "simple mask": "simple_mask",
    "simple_mask": "simple_mask",
    "nrb": "NRB",
    "nonrebreather": "NRB",
    "non-rebreather": "NRB",
    "non rebreather": "NRB",
    "bvm": "BVM",
    "bag valve mask": "BVM",
    "bag_valve_mask": "BVM",
    "bipap": "BiPAP",
    "hfnc": "HFNC",
    "high flow nasal cannula": "HFNC",
    "vent": "vent",
    "ventilator": "vent",
}

MEDICINE_LIKE_ACTIONS = {
    "vasopressor",
    "vasodilator",
    "inotrope",
    "rate_control",
    "chronotropic_agent",
    "antiarrhythmic",
    "antidote",
    "membrane_stabilization",
    "hyperkalemia_shift",
    "electrolyte_correction",
    "corticosteroid",
    "anticonvulsant",
    "anticoagulation",
    "analgesia",
    "sedation",
    "bronchodilator",
    "anaphylaxis_treatment",
    "prostaglandin_infusion",
    "osmotherapy",
    "airway_medication",
}

PROCEDURE_LIKE_ACTIONS = {
    "tube_thoracostomy",
    "surgical_airway",
    "needle_decompression",
    "uterine_exploration",
    "obstetric_delivery",
    "transcutaneous_pacing",
    "synchronized_cardioversion",
    "oral_airway",
}

NO_IMMEDIATE_MEDICINE_EFFECT = {
    "airway_medication",
    "membrane_stabilization",
    "hyperkalemia_shift",
    "electrolyte_correction",
    "corticosteroid",
    "anticonvulsant",
    "anticoagulation",
    "prostaglandin_infusion",
    "osmotherapy",
}


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def clamp_vitals(vitals: dict[str, Any]) -> dict[str, Any]:
    """Clamp present canonical vitals to physiologic bounds."""

    for vital, (lower, upper) in VITAL_BOUNDS.items():
        value = vitals.get(vital)
        if _is_number(value):
            vitals[vital] = clamp(value, lower, upper)
    return vitals


def move_toward(value: Any, target: float, max_delta: float) -> Any:
    if not _is_number(value):
        return value
    delta = abs(max_delta)
    if value < target:
        return min(target, value + delta)
    if value > target:
        return max(target, value - delta)
    return value


def inc_spo2(vitals: dict[str, Any], delta: float, ceiling: float) -> None:
    value = vitals.get("O2Sat")
    if _is_number(value) and value < ceiling:
        vitals["O2Sat"] = min(ceiling, value + delta)


def dec_spo2(vitals: dict[str, Any], delta: float, floor: float) -> None:
    value = vitals.get("O2Sat")
    if _is_number(value) and value > floor:
        vitals["O2Sat"] = max(floor, value - delta)


def inc_bp(vitals: dict[str, Any], sys_delta: float, dia_delta: float) -> None:
    _add_if_present(vitals, "BP_sys", sys_delta)
    _add_if_present(vitals, "BP_dia", dia_delta)


def inc_bp_if_low(
    vitals: dict[str, Any],
    sys_delta: float,
    dia_delta: float,
    threshold: float = 100,
) -> None:
    if _is_number(vitals.get("BP_sys")) and vitals["BP_sys"] < threshold:
        inc_bp(vitals, sys_delta, dia_delta)


def dec_bp(vitals: dict[str, Any], sys_delta: float, dia_delta: float) -> None:
    _add_if_present(vitals, "BP_sys", -abs(sys_delta))
    _add_if_present(vitals, "BP_dia", -abs(dia_delta))


def move_hr_toward(vitals: dict[str, Any], target: float, max_delta: float) -> None:
    value = vitals.get("HR")
    if _is_number(value):
        vitals["HR"] = move_toward(value, target, max_delta)


def dec_hr_if_high(
    vitals: dict[str, Any],
    delta: float,
    threshold: float = 110,
    floor: float | None = None,
) -> None:
    value = vitals.get("HR")
    if _is_number(value) and value > threshold:
        next_value = value - abs(delta)
        vitals["HR"] = max(floor, next_value) if floor is not None else next_value


def inc_hr_if_low(
    vitals: dict[str, Any],
    delta: float,
    threshold: float = 60,
    ceiling: float | None = None,
) -> None:
    value = vitals.get("HR")
    if _is_number(value) and value < threshold:
        next_value = value + abs(delta)
        vitals["HR"] = min(ceiling, next_value) if ceiling is not None else next_value


def dec_rr_if_high(
    vitals: dict[str, Any],
    delta: float,
    threshold: float = 20,
    floor: float | None = None,
) -> None:
    value = vitals.get("RR")
    if _is_number(value) and value > threshold:
        next_value = value - abs(delta)
        vitals["RR"] = max(floor, next_value) if floor is not None else next_value


def set_if_present(vitals: dict[str, Any], vital: str, value: Any) -> None:
    if vitals.get(vital) is not None:
        vitals[vital] = value


class RuleBasedEngine:
    """Conservative deterministic vital-sign and feature transition engine."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def predict(self, engine_input: dict[str, Any]) -> EngineOutput:
        before = engine_input.get("before", {})
        before_vitals = before.get("vitals", {}) if isinstance(before, dict) else {}
        vitals = _copy_canonical_vitals(before_vitals)
        before_features = (
            before.get("features", {}) if isinstance(before, dict) else {}
        )
        features = _copy_features(before_features)

        action = engine_input.get("action", {})
        if not isinstance(action, dict):
            action = {}
        kind_hint = action.get("kind_hint")
        params = _params(action.get("params"))

        handled = self._dispatch(vitals, kind_hint, params)
        _apply_feature_rules(features, kind_hint, params)
        if not handled and self.strict:
            raise ValueError(f"Unknown action kind_hint: {kind_hint!r}")

        clamp_vitals(vitals)
        return {"prediction": {"vitals": vitals, "features": features}}

    def _dispatch(
        self,
        vitals: dict[str, Any],
        kind_hint: Any,
        params: dict[str, Any],
    ) -> bool:
        if kind_hint == "oxygen_support":
            _apply_oxygen_support(vitals, params)
        elif kind_hint == "airway_management":
            _apply_airway_management(vitals, params)
        elif kind_hint == "fluid_bolus":
            _apply_fluid_bolus(vitals, params)
        elif kind_hint == "blood_transfusion":
            _apply_blood_transfusion(vitals, params)
        elif kind_hint in MEDICINE_LIKE_ACTIONS:
            _apply_medicine_like(vitals, kind_hint)
        elif kind_hint in PROCEDURE_LIKE_ACTIONS:
            _apply_procedure_like(vitals, kind_hint)
        elif kind_hint in {"no_action", "time_passage"}:
            _apply_no_action(vitals, params)
        else:
            return False
        return True


def predict_features(engine_input: dict[str, Any]) -> dict[str, Any]:
    """Predict deterministic non-vital feature changes from action structure."""

    before = engine_input.get("before", {})
    before_features = (
        before.get("features", {}) if isinstance(before, dict) else {}
    )
    features = _copy_features(before_features)

    action = engine_input.get("action", {})
    if not isinstance(action, dict):
        action = {}
    _apply_feature_rules(
        features,
        action.get("kind_hint"),
        _params(action.get("params")),
    )
    return features


def _apply_oxygen_support(vitals: dict[str, Any], params: dict[str, Any]) -> None:
    old_spo2 = vitals.get("O2Sat")
    device = _normalized_text(params.get("oxygen_device"))
    peep_used = params.get("PEEP_used") is True
    peep_value = params.get("PEEP_cmH2O")
    if _is_number(peep_value) and peep_value > 0:
        peep_used = True

    if device is None:
        target, max_delta = 92, 3
    elif device in {"nasal_cannula", "nasal cannula", "nc"}:
        target, max_delta = 92, 4
    elif device in {"simple_mask", "simple mask"}:
        target, max_delta = 94, 5
    elif device in {"nrb", "nonrebreather", "non-rebreather", "non rebreather"}:
        target, max_delta = 94, 8
    elif device in {"bvm", "bag_valve_mask", "bag valve mask"}:
        target, max_delta = (96, 8) if peep_used else (95, 6)
    else:
        target, max_delta = 94, 5

    vitals["O2Sat"] = move_toward(old_spo2, target, max_delta)
    if _is_number(old_spo2) and _is_number(vitals.get("O2Sat")):
        if vitals["O2Sat"] > old_spo2:
            dec_rr_if_high(vitals, 2, threshold=24)


def _apply_airway_management(vitals: dict[str, Any], params: dict[str, Any]) -> None:
    procedure = _normalized_text(params.get("procedure"))
    stage = _normalized_text(params.get("stage"))
    completed_intubation = (
        params.get("intubated") is True
        or (procedure == "intubation" and stage == "completed")
    )

    if not completed_intubation:
        return

    set_if_present(vitals, "RR", 12)
    vitals["O2Sat"] = move_toward(vitals.get("O2Sat"), 95, 10)
    dec_hr_if_high(vitals, 15, threshold=110, floor=60)


def _apply_fluid_bolus(vitals: dict[str, Any], params: dict[str, Any]) -> None:
    before_bp_sys = vitals.get("BP_sys")
    if _is_number(before_bp_sys):
        if before_bp_sys < 90:
            sys_delta, dia_delta = 12, 6
        elif before_bp_sys < 110:
            sys_delta, dia_delta = 8, 4
        else:
            sys_delta, dia_delta = 0, 0

        volume_ml = params.get("volume_ml")
        if sys_delta and _is_number(volume_ml) and volume_ml >= 1000:
            sys_delta += 3
            dia_delta += 2

        if sys_delta:
            inc_bp(vitals, sys_delta, dia_delta)

    if (
        _is_number(vitals.get("HR"))
        and _is_number(before_bp_sys)
        and before_bp_sys < 100
        and vitals["HR"] > 110
    ):
        vitals["HR"] -= 5


def _apply_blood_transfusion(vitals: dict[str, Any], params: dict[str, Any]) -> None:
    effect_scale = 1.0
    units = params.get("units")
    if _is_number(units) and units >= 2:
        effect_scale = 1.5
    if params.get("strategy") == "massive_transfusion":
        effect_scale = 2.0

    if _is_number(vitals.get("BP_sys")) and vitals["BP_sys"] < 100:
        vitals["BP_sys"] += int(10 * effect_scale)
    if _is_number(vitals.get("BP_dia")) and vitals["BP_dia"] < 65:
        vitals["BP_dia"] += int(5 * effect_scale)
    if _is_number(vitals.get("HR")) and vitals["HR"] > 110:
        vitals["HR"] -= int(5 * effect_scale)


def _apply_medicine_like(vitals: dict[str, Any], kind_hint: str) -> None:
    if kind_hint == "vasopressor":
        inc_bp_if_low(vitals, 15, 8, threshold=100)
    elif kind_hint == "vasodilator":
        if _is_number(vitals.get("BP_sys")) and vitals["BP_sys"] > 140:
            dec_bp(vitals, 15, 8)
    elif kind_hint == "inotrope":
        inc_bp(vitals, 8, 4)
        _add_if_present(vitals, "HR", 5)
    elif kind_hint == "rate_control":
        move_hr_toward(vitals, 90, 25)
    elif kind_hint == "chronotropic_agent":
        _add_if_present(vitals, "HR", 15)
    elif kind_hint == "antiarrhythmic":
        move_hr_toward(vitals, 90, 20)
    elif kind_hint == "bronchodilator":
        inc_spo2(vitals, 3, 96)
        dec_rr_if_high(vitals, 3, threshold=20)
    elif kind_hint == "sedation":
        dec_rr_if_high(vitals, 2, threshold=12)
        dec_bp(vitals, 5, 3)
    elif kind_hint == "analgesia":
        dec_hr_if_high(vitals, 5, threshold=100)
        dec_rr_if_high(vitals, 2, threshold=20)
    elif kind_hint == "anaphylaxis_treatment":
        inc_bp(vitals, 15, 8)
        inc_spo2(vitals, 5, 96)
        dec_rr_if_high(vitals, 4, threshold=20)
    elif kind_hint == "antidote":
        dec_hr_if_high(vitals, 10, threshold=120)
        inc_bp_if_low(vitals, 10, 5, threshold=90)
    elif kind_hint in NO_IMMEDIATE_MEDICINE_EFFECT:
        return


def _apply_procedure_like(vitals: dict[str, Any], kind_hint: str) -> None:
    if kind_hint == "needle_decompression":
        inc_spo2(vitals, 6, 94)
        inc_bp_if_low(vitals, 10, 5, threshold=100)
        dec_hr_if_high(vitals, 8, threshold=110)
    elif kind_hint == "tube_thoracostomy":
        inc_spo2(vitals, 5, 95)
        dec_rr_if_high(vitals, 4, threshold=20)
        inc_bp_if_low(vitals, 8, 4, threshold=100)
    elif kind_hint == "surgical_airway":
        set_if_present(vitals, "RR", 12)
        vitals["O2Sat"] = move_toward(vitals.get("O2Sat"), 95, 8)
        dec_hr_if_high(vitals, 10, threshold=110)
    elif kind_hint == "oral_airway":
        inc_spo2(vitals, 3, 94)
    elif kind_hint == "transcutaneous_pacing":
        hr = vitals.get("HR")
        if _is_number(hr) and hr < 60:
            vitals["HR"] = min(80, hr + 25)
        inc_bp_if_low(vitals, 10, 5, threshold=100)
    elif kind_hint == "synchronized_cardioversion":
        move_hr_toward(vitals, 90, 40)
        inc_bp_if_low(vitals, 8, 4, threshold=100)
    elif kind_hint in {"uterine_exploration", "obstetric_delivery"}:
        return


def _apply_no_action(vitals: dict[str, Any], params: dict[str, Any]) -> None:
    elapsed = params.get("elapsed_min")
    if not _is_number(elapsed):
        elapsed = 5
    scale = 0 if elapsed == 0 else clamp(elapsed / 5, 0.5, 2.0)

    if _is_number(vitals.get("O2Sat")) and vitals["O2Sat"] < 90:
        dec_spo2(vitals, int(3 * scale), floor=50)

    if _is_number(vitals.get("RR")) and _is_number(vitals.get("O2Sat")):
        if vitals["RR"] > 30 and vitals["O2Sat"] < 90:
            # Falling RR here reflects hypoxemic respiratory fatigue, not improvement.
            vitals["RR"] = max(24, vitals["RR"] - int(4 * scale))
        elif vitals["RR"] < 10 and vitals["O2Sat"] < 90:
            vitals["RR"] = max(4, vitals["RR"] - int(1 * scale))

    shock_condition = _is_number(vitals.get("BP_sys")) and vitals["BP_sys"] < 90
    severe_tachycardia_condition = _is_number(vitals.get("HR")) and vitals["HR"] > 140
    if shock_condition:
        vitals["BP_sys"] = max(40, vitals["BP_sys"] - int(5 * scale))
        if _is_number(vitals.get("BP_dia")):
            vitals["BP_dia"] = max(20, vitals["BP_dia"] - int(3 * scale))
        if _is_number(vitals.get("HR")):
            vitals["HR"] = min(180, vitals["HR"] + int(5 * scale))
    elif severe_tachycardia_condition:
        if _is_number(vitals.get("BP_sys")):
            vitals["BP_sys"] -= int(3 * scale)


def _apply_feature_rules(
    features: dict[str, Any],
    kind_hint: Any,
    params: dict[str, Any],
) -> None:
    if kind_hint == "oxygen_support":
        _apply_oxygen_support_features(features, params)
    elif kind_hint == "airway_management":
        _apply_airway_management_features(features, params)
    elif kind_hint == "surgical_airway":
        _apply_secured_airway_features(features)
    elif kind_hint == "synchronized_cardioversion":
        features["rhythm"] = "sinus"
    elif kind_hint == "transcutaneous_pacing":
        features["rhythm"] = "paced"
    elif kind_hint in {"antiarrhythmic", "membrane_stabilization"}:
        features["rhythm"] = "sinus"


def _apply_oxygen_support_features(
    features: dict[str, Any],
    params: dict[str, Any],
) -> None:
    device = _canonical_oxygen_device(params.get("oxygen_device"))
    fio2 = params.get("FiO2")
    peep_value = params.get("PEEP_cmH2O")
    peep_used = params.get("PEEP_used") is True
    if _is_number(peep_value) and peep_value > 0:
        peep_used = True
    if device == "BiPAP":
        peep_used = True

    if device is not None:
        features["oxygen_device"] = device
        features["FiO2"] = (
            fio2 if _is_number(fio2) else OXYGEN_DEVICE_DEFAULT_FIO2.get(device)
        )
        if device in {"BVM", "BiPAP", "vent"}:
            features["vent"] = True
        else:
            features["vent"] = False
        if device == "vent":
            features["intubated"] = True
    elif _is_number(fio2):
        features["FiO2"] = fio2

    if _is_number(peep_value):
        features["PEEP_cmH2O"] = peep_value
    elif peep_used:
        features["PEEP_cmH2O"] = features.get("PEEP_cmH2O") or 5
    elif params.get("PEEP_used") is False:
        features["PEEP_cmH2O"] = None


def _apply_airway_management_features(
    features: dict[str, Any],
    params: dict[str, Any],
) -> None:
    procedure = _normalized_text(params.get("procedure"))
    stage = _normalized_text(params.get("stage"))
    completed_intubation = (
        params.get("intubated") is True
        or (procedure == "intubation" and stage == "completed")
    )

    if params.get("intubated") is False:
        features["intubated"] = False
        if stage != "completed":
            return

    if completed_intubation:
        _apply_secured_airway_features(features)
        if params.get("rsi_medication_used") is True:
            features["neuro_status"] = "intubated and sedated"


def _apply_secured_airway_features(features: dict[str, Any]) -> None:
    features["intubated"] = True
    features["vent"] = True
    features["oxygen_device"] = "vent"
    features["FiO2"] = 1.0
    features["PEEP_cmH2O"] = features.get("PEEP_cmH2O") or 5


def _copy_canonical_vitals(before_vitals: Any) -> dict[str, Any]:
    if not isinstance(before_vitals, dict):
        before_vitals = {}
    return {vital: deepcopy(before_vitals.get(vital)) for vital in CANONICAL_VITAL_KEYS}


def _copy_features(before_features: Any) -> dict[str, Any]:
    return deepcopy(before_features) if isinstance(before_features, dict) else {}


def _params(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _canonical_oxygen_device(value: Any) -> str | None:
    normalized = _normalized_text(value)
    if normalized is None:
        return None
    return OXYGEN_DEVICE_ALIASES.get(normalized, str(value).strip())


def _add_if_present(vitals: dict[str, Any], vital: str, delta: float) -> None:
    value = vitals.get(vital)
    if _is_number(value):
        vitals[vital] = value + delta


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "MEDICINE_LIKE_ACTIONS",
    "OXYGEN_DEVICE_DEFAULT_FIO2",
    "PROCEDURE_LIKE_ACTIONS",
    "RuleBasedEngine",
    "clamp_vitals",
    "dec_bp",
    "dec_hr_if_high",
    "dec_rr_if_high",
    "dec_spo2",
    "inc_bp",
    "inc_bp_if_low",
    "inc_hr_if_low",
    "inc_spo2",
    "move_hr_toward",
    "move_toward",
    "predict_features",
    "set_if_present",
]
