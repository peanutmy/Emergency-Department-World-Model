"""Two-stage direction-first hybrid engine for transition-pair evaluation."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Protocol

from .common import (
    CANONICAL_VITAL_KEYS,
    DIRECTION_DEAD_ZONES,
    EngineOutput,
    direction as classify_direction,
    sanitize_case_context,
)
from .rule_based import RuleBasedEngine, clamp_vitals


class LLMClient(Protocol):
    """Minimal client interface used by each LLM stage."""

    def complete(self, prompt: str) -> str:
        """Return model response text for one prompt."""


DIRECTION_LABELS = ("increase", "decrease", "stable")
STAGE1_PROMPT_PROFILES = (
    "original",
    "calibrated_v2",
    "calibrated_v3",
    "calibrated_v4",
)
MIN_DIRECTIONAL_MAGNITUDES = {
    "HR": 6.0,
    "BP_sys": 6.0,
    "BP_dia": 6.0,
    "RR": 3.0,
    "O2Sat": 3.0,
    "T": 0.3,
}

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FORBIDDEN_PROMPT_KEYS = {
    "label",
    "modifier_text",
    "source",
    "supporting_findings",
    "target",
}


class DirectionFirstHybridEngine:
    """Choose vital directions first, then estimate unsigned magnitudes."""

    def __init__(
        self,
        rule_engine: RuleBasedEngine,
        direction_client: LLMClient,
        magnitude_client: LLMClient,
        stage1_prompt_profile: str = "original",
    ) -> None:
        if stage1_prompt_profile not in STAGE1_PROMPT_PROFILES:
            raise ValueError(
                f"Unsupported Stage 1 prompt profile: {stage1_prompt_profile!r}"
            )
        self.rule_engine = rule_engine
        self.direction_client = direction_client
        self.magnitude_client = magnitude_client
        self.stage1_prompt_profile = stage1_prompt_profile

    def predict(
        self,
        engine_input: dict[str, Any],
        target_vital_names: Iterable[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> EngineOutput:
        """Predict post-transition vitals with a direction-locked second stage."""

        requested_vitals = _requested_vitals(target_vital_names)
        before_vitals = _before_vitals(engine_input)
        rule_output = self.rule_engine.predict(engine_input)
        rule_prediction = _mapping(rule_output.get("prediction"))
        rule_vitals = _canonical_vitals(rule_prediction.get("vitals", {}))
        rule_features = deepcopy(_mapping(rule_prediction.get("features")))

        direction_prompt = build_direction_prompt(
            engine_input=engine_input,
            target_vital_names=requested_vitals,
            examples=examples,
            prompt_profile=self.stage1_prompt_profile,
        )
        direction_call = _call_client(self.direction_client, direction_prompt)
        (
            llm_directions,
            direction_reasoning,
            direction_parse_error,
        ) = parse_llm_directions(direction_call["text"])
        selected_directions, direction_sources = _resolve_directions(
            requested_vitals=requested_vitals,
            before_vitals=before_vitals,
            rule_vitals=rule_vitals,
            llm_directions=llm_directions,
        )

        magnitude_prompt = build_magnitude_prompt(
            engine_input=engine_input,
            rule_prediction_vitals=rule_vitals,
            selected_directions=selected_directions,
            target_vital_names=requested_vitals,
            examples=examples,
        )
        magnitude_call = _call_client(self.magnitude_client, magnitude_prompt)
        (
            llm_magnitudes,
            magnitude_reasoning,
            magnitude_parse_error,
        ) = parse_llm_magnitudes(magnitude_call["text"])

        (
            final_vitals,
            resolved_magnitudes,
            magnitude_sources,
            effective_directions,
            residuals,
        ) = compose_directional_vitals(
            before_vitals=before_vitals,
            rule_prediction_vitals=rule_vitals,
            selected_directions=selected_directions,
            llm_magnitudes=llm_magnitudes,
            target_vital_names=requested_vitals,
        )

        return {
            "prediction": {
                "vitals": final_vitals,
                "features": rule_features,
            },
            "metadata": {
                "engine": "direction_first_hybrid",
                "stage1_prompt_profile": self.stage1_prompt_profile,
                "rule_prediction": {
                    "vitals": rule_vitals,
                    "features": rule_features,
                },
                "selected_directions": selected_directions,
                "direction_sources": direction_sources,
                "effective_directions": effective_directions,
                "resolved_magnitudes": resolved_magnitudes,
                "magnitude_sources": magnitude_sources,
                "computed_residuals": residuals,
                "few_shot_examples": _example_references(examples),
                "direction_stage": {
                    "llm_directions": llm_directions,
                    "llm_reasoning": direction_reasoning,
                    "llm_raw_response": direction_call["text"],
                    "llm_parse_error": direction_parse_error,
                    "llm_api_error": direction_call["api_error"],
                    "llm_error": direction_call["error"],
                    "model": _client_attr(self.direction_client, "model"),
                    "prompt_profile": self.stage1_prompt_profile,
                },
                "magnitude_stage": {
                    "llm_magnitudes": llm_magnitudes,
                    "llm_reasoning": magnitude_reasoning,
                    "llm_raw_response": magnitude_call["text"],
                    "llm_parse_error": magnitude_parse_error,
                    "llm_api_error": magnitude_call["api_error"],
                    "llm_error": magnitude_call["error"],
                    "model": _client_attr(self.magnitude_client, "model"),
                },
            },
        }


def build_direction_prompt(
    *,
    engine_input: dict[str, Any],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
    prompt_profile: str = "original",
) -> str:
    """Build Stage 1 prompt without exposing the rule-based prediction."""

    if prompt_profile == "calibrated_v2":
        return build_calibrated_direction_prompt(
            engine_input=engine_input,
            target_vital_names=target_vital_names,
            examples=examples,
        )
    if prompt_profile == "calibrated_v3":
        return build_calibrated_v3_direction_prompt(
            engine_input=engine_input,
            target_vital_names=target_vital_names,
            examples=examples,
        )
    if prompt_profile == "calibrated_v4":
        return build_calibrated_v4_direction_prompt(
            engine_input=engine_input,
            target_vital_names=target_vital_names,
            examples=examples,
        )
    if prompt_profile != "original":
        raise ValueError(
            f"Unsupported Stage 1 prompt profile: {prompt_profile!r}"
        )

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    return f"""You are Stage 1 of a vital-sign transition engine for an educational emergency medicine simulation.

Decide only the direction of change from the pre-transition value for each requested vital.
Do not estimate final values or magnitudes.

Direction labels:
- increase: change is greater than the vital's stable dead zone.
- decrease: change is less than the negative stable dead zone.
- stable: absolute change is within the stable dead zone, including the boundary.

Stable dead zones:
{_json({vital: DIRECTION_DEAD_ZONES[vital] for vital in target_vital_names})}

Rules:
- Use only the supplied case context, current state, and action.
- Do not infer hidden diagnoses or unreleased findings.
- A no_action transition may represent ongoing deterioration or scenario progression; do not default it to stable.
- Judge each requested vital independently.
- Return directions for requested vitals only.
- Include one concise clinical rationale per requested vital.
- Output JSON only.

{_render_direction_examples(examples)}Case:
{_json(_case_block(engine_input))}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub(deepcopy(before.get("features", {}))))}

Action:
{_json(_action_block(action))}

Requested vitals:
{_json(target_vital_names)}

Return JSON only:
{_json({"directions": _null_template(target_vital_names), "reasoning": _null_template(target_vital_names)})}"""


def build_calibrated_direction_prompt(
    *,
    engine_input: dict[str, Any],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build calibrated Stage 1 prompt with an explicit stable/change gate."""

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    return f"""You are Stage 1 of a vital-sign transition engine for an educational emergency medicine simulation.

Decide only the evaluated direction of change from the pre-transition value for each requested vital.
Do not estimate final values or magnitudes.

Direction labels:
- increase: change is greater than the vital's stable dead zone.
- decrease: change is less than the negative stable dead zone.
- stable: absolute change is within the stable dead zone, including the boundary.

Stable dead zones:
{_json({vital: DIRECTION_DEAD_ZONES[vital] for vital in target_vital_names})}

Calibration objective:
- The label represents a thresholded observed change during this transition interval, not merely the sign of a physiologically plausible effect.
- Treat stable as a positive clinical prediction that requires active consideration.
- A plausible tendency alone is not enough to select increase or decrease.

Decision policy for each requested vital:
1. Use only the supplied case context, current state, action, and transition interval.
2. Decide whether the supplied evidence supports a change during this interval.
3. Decide whether the most likely change clearly exceeds the vital-specific stable dead zone.
4. Select increase or decrease only when the evidence supports both the direction and a change larger than the dead zone.
5. Select stable when the likely effect is small, delayed, indirect, uncertain near the threshold, or expected to remain within the dead zone.
6. Do not assume that an intervention changes every physiologically related vital. Require a direct and sufficiently strong immediate effect for each requested vital.
7. no_action does not automatically mean stable or deterioration. Select deterioration only when the current instability, explicit scenario progression, or supplied features support a clinically meaningful change during the interval.
8. Do not infer hidden diagnoses, complications, phases, or unreleased findings.

Rationale requirement:
- Give one concise rationale per requested vital.
- For increase or decrease, identify the supplied evidence and explain why the effect should exceed the dead zone during this interval.
- If that threshold justification is not supported, select stable.
- Output JSON only.

{_render_direction_examples(examples)}Case:
{_json(_case_block(engine_input))}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub(deepcopy(before.get("features", {}))))}

Action:
{_json(_action_block(action))}

Requested vitals:
{_json(target_vital_names)}

Return JSON only:
{_json({"directions": _null_template(target_vital_names), "reasoning": _null_template(target_vital_names)})}"""


def build_calibrated_v3_direction_prompt(
    *,
    engine_input: dict[str, Any],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build a balanced Stage 1 prompt without a stable-class default."""

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    return f"""You are Stage 1 of a vital-sign transition engine for an educational emergency medicine simulation.

Decide only the evaluated direction of change from the pre-transition value for each requested vital.
Do not estimate final values or magnitudes.

Direction labels:
- increase: change is greater than the vital's stable dead zone.
- decrease: change is less than the negative stable dead zone.
- stable: absolute change is within the stable dead zone, including the boundary.

Stable dead zones:
{_json({vital: DIRECTION_DEAD_ZONES[vital] for vital in target_vital_names})}

Calibration objective:
- The label represents the most likely thresholded observed change during this transition interval, not merely the sign of a physiologically plausible effect.
- No direction category is the default.
- Uncertainty alone is not evidence of stability, and a plausible mechanism alone is not evidence that the dead zone will be crossed.
- Choose the single most likely category for each requested vital.

Decision policy for each requested vital:
1. Use only the supplied case context, current state, action, and transition interval.
2. Identify the strongest supplied evidence about the current trajectory and the action's immediate effect.
3. Estimate whether the most likely absolute change is within or beyond the vital-specific stable dead zone.
4. Select stable when the most likely change is within the dead zone, including its boundary. Select increase or decrease when the most likely change crosses the dead zone in that direction; certainty is not required.
5. Explicit instability or progression can support a non-stable direction even when the full mechanism is not supplied.
6. For an intervention, distinguish a direct, immediate, sufficiently strong effect from an indirect, delayed, or small effect that is likely to remain within the dead zone.
7. no_action has no default direction. Use the supplied current instability, features, and explicit progression. Do not require the literal word "worsening" when the supplied state supports an ongoing trajectory.
8. When the outcome is close to the threshold, choose the category matching the most likely magnitude; do not automatically resolve uncertainty to stable.
9. Do not infer hidden diagnoses, complications, phases, or unreleased findings.

Rationale requirement:
- Give one concise rationale per requested vital.
- Cite the supplied evidence and state whether the most likely change crosses that vital's dead zone during this interval.
- Output JSON only.

{_render_direction_examples(examples)}Case:
{_json(_case_block(engine_input))}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub(deepcopy(before.get("features", {}))))}

Action:
{_json(_action_block(action))}

Requested vitals:
{_json(target_vital_names)}

Return JSON only:
{_json({"directions": _null_template(target_vital_names), "reasoning": _null_template(target_vital_names)})}"""


def build_calibrated_v4_direction_prompt(
    *,
    engine_input: dict[str, Any],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build Stage 1 prompt with explicit workflow and threshold semantics."""

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    return f"""You are Stage 1 of a two-stage vital-sign transition predictor.

Workflow:
- Stage 1 selects the evaluated direction from the pre-transition value for each requested vital.
- Stage 2 receives the selected direction as fixed and estimates only the magnitude; it cannot change the direction.
- A direction error from Stage 1 cannot be corrected in Stage 2.
- Do not output exact final values or magnitudes. Judge only whether the most likely change crosses the evaluation threshold.

The stable dead zone is an evaluation threshold around each vital's pre-transition value, not a normal clinical range.
A change within or exactly on the dead-zone boundary is stable. Only a change beyond the boundary is increase or decrease.

Direction labels:
- increase: change is greater than the vital's stable dead zone.
- decrease: change is less than the negative stable dead zone.
- stable: absolute change is within the stable dead zone, including the boundary.

Stable dead zones:
{_json({vital: DIRECTION_DEAD_ZONES[vital] for vital in target_vital_names})}

Calibration objective:
- The label represents the most likely thresholded observed change during this transition interval, not merely the sign of a physiologically plausible effect.
- No direction category is the default.
- Uncertainty alone is not evidence of stability, and a plausible mechanism alone is not evidence that the dead zone will be crossed.
- Choose the single most likely category for each requested vital.

Decision policy for each requested vital:
1. Use only the supplied case context, current state, action, and transition interval.
2. Identify the strongest supplied evidence about the current trajectory and the action's immediate effect.
3. Estimate whether the most likely absolute change is within or beyond the vital-specific stable dead zone.
4. Select stable when the most likely change is within the dead zone, including its boundary. Select increase or decrease when the most likely change crosses the dead zone in that direction; certainty is not required.
5. Explicit instability or progression can support a non-stable direction even when the full mechanism is not supplied.
6. For an intervention, distinguish a direct, immediate, sufficiently strong effect from an indirect, delayed, or small effect that is likely to remain within the dead zone.
7. no_action has no default direction. Use the supplied current instability, features, and explicit progression. Do not require the literal word "worsening" when the supplied state supports an ongoing trajectory.
8. When the outcome is close to the threshold, choose the category matching the most likely magnitude; do not automatically resolve uncertainty to stable.
9. Do not infer hidden diagnoses, complications, phases, or unreleased findings.

Rationale requirement:
- Give one concise rationale per requested vital.
- Cite the supplied evidence and state whether the most likely change crosses that vital's dead zone during this interval.
- Output JSON only.

{_render_direction_examples(examples)}Case:
{_json(_case_block(engine_input))}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub(deepcopy(before.get("features", {}))))}

Action:
{_json(_action_block(action))}

Requested vitals:
{_json(target_vital_names)}

Return JSON only:
{_json({"directions": _null_template(target_vital_names), "reasoning": _null_template(target_vital_names)})}"""


def build_magnitude_prompt(
    *,
    engine_input: dict[str, Any],
    rule_prediction_vitals: dict[str, Any],
    selected_directions: dict[str, str],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build Stage 2 prompt with locked directions and a rule magnitude reference."""

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    minimums = {
        vital: MIN_DIRECTIONAL_MAGNITUDES[vital]
        for vital in target_vital_names
    }
    return f"""You are Stage 2 of a vital-sign transition engine for an educational emergency medicine simulation.

The direction for each requested vital has already been selected and is fixed.
Do not reconsider or change any direction.

For increase or decrease, return a nonnegative magnitude measured from the pre-transition value.
For stable, return null or 0; the engine will preserve the pre-transition value.
The deterministic rule-based prediction is a rough magnitude reference only and may be wrong.

Minimum magnitudes that produce a non-stable evaluated direction:
{_json(minimums)}

Rules:
- Use only the supplied case context, current state, action, fixed directions, and rule prediction.
- Do not infer hidden diagnoses or unreleased findings.
- Respect the fixed direction even when the rule prediction points another way.
- Keep the resulting vital physiologically plausible.
- Return magnitudes for requested vitals only.
- Include one concise rationale per requested vital.
- Output JSON only.

{_render_magnitude_examples(examples)}Case:
{_json(_case_block(engine_input))}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub(deepcopy(before.get("features", {}))))}

Action:
{_json(_action_block(action))}

Fixed directions:
{_json(selected_directions)}

Rule-based prediction:
{_json(_canonical_vitals(rule_prediction_vitals))}

Requested vitals:
{_json(target_vital_names)}

Return JSON only:
{_json({"magnitudes": _null_template(target_vital_names), "reasoning": _null_template(target_vital_names)})}"""


def parse_llm_directions(
    text: str,
) -> tuple[dict[str, str | None], dict[str, Any], str | None]:
    """Parse per-vital direction labels and concise rationales."""

    parsed, parse_error = _parse_json_object(text)
    if parsed is None:
        return {}, {}, parse_error
    raw = parsed.get("directions")
    if not isinstance(raw, dict):
        return {}, _extract_reasoning(parsed), "Missing directions object"

    directions: dict[str, str | None] = {}
    invalid: list[str] = []
    for vital in CANONICAL_VITAL_KEYS:
        if vital not in raw:
            continue
        value = _normalize_direction(raw.get(vital))
        directions[vital] = value
        if value is None and raw.get(vital) is not None:
            invalid.append(vital)
    error = f"Invalid direction labels for: {', '.join(invalid)}" if invalid else None
    return directions, _extract_reasoning(parsed), error


def parse_llm_magnitudes(
    text: str,
) -> tuple[dict[str, float | None], dict[str, Any], str | None]:
    """Parse nonnegative per-vital magnitudes and concise rationales."""

    parsed, parse_error = _parse_json_object(text)
    if parsed is None:
        return {}, {}, parse_error
    raw = parsed.get("magnitudes")
    if not isinstance(raw, dict):
        return {}, _extract_reasoning(parsed), "Missing magnitudes object"

    magnitudes: dict[str, float | None] = {}
    invalid: list[str] = []
    for vital in CANONICAL_VITAL_KEYS:
        if vital not in raw:
            continue
        value = raw.get(vital)
        if value is None:
            magnitudes[vital] = None
        elif _is_number(value) and float(value) >= 0:
            magnitudes[vital] = float(value)
        else:
            magnitudes[vital] = None
            invalid.append(vital)
    error = f"Invalid magnitudes for: {', '.join(invalid)}" if invalid else None
    return magnitudes, _extract_reasoning(parsed), error


def compose_directional_vitals(
    *,
    before_vitals: dict[str, Any],
    rule_prediction_vitals: dict[str, Any],
    selected_directions: dict[str, str],
    llm_magnitudes: dict[str, float | None],
    target_vital_names: Iterable[str],
) -> tuple[
    dict[str, Any],
    dict[str, float | None],
    dict[str, str],
    dict[str, str | None],
    dict[str, float | None],
]:
    """Apply locked directions and resolved magnitudes to the before state."""

    final_vitals = _canonical_vitals(rule_prediction_vitals)
    resolved_magnitudes: dict[str, float | None] = {}
    magnitude_sources: dict[str, str] = {}
    requested_vitals = _requested_vitals(target_vital_names)

    for vital in requested_vitals:
        before_value = before_vitals.get(vital)
        if not _is_number(before_value):
            resolved_magnitudes[vital] = None
            magnitude_sources[vital] = "unavailable_before_value"
            continue

        selected_direction = selected_directions.get(vital, "stable")
        if selected_direction == "stable":
            final_vitals[vital] = float(before_value)
            resolved_magnitudes[vital] = 0.0
            magnitude_sources[vital] = "deterministic_stable"
            continue

        llm_magnitude = llm_magnitudes.get(vital)
        if _is_number(llm_magnitude) and float(llm_magnitude) >= 0:
            magnitude = float(llm_magnitude)
            source = "llm"
        else:
            rule_value = rule_prediction_vitals.get(vital)
            magnitude = (
                abs(float(rule_value) - float(before_value))
                if _is_number(rule_value)
                else 0.0
            )
            source = "rule_absolute_delta"

        minimum = MIN_DIRECTIONAL_MAGNITUDES[vital]
        if magnitude <= DIRECTION_DEAD_ZONES[vital] or math.isclose(
            magnitude,
            DIRECTION_DEAD_ZONES[vital],
        ):
            magnitude = minimum
            source = f"{source}_dead_zone_floor"

        signed_magnitude = magnitude if selected_direction == "increase" else -magnitude
        final_vitals[vital] = float(before_value) + signed_magnitude
        resolved_magnitudes[vital] = magnitude
        magnitude_sources[vital] = source

    final_vitals = clamp_vitals(final_vitals)
    effective_directions: dict[str, str | None] = {}
    residuals: dict[str, float | None] = {}
    for vital in requested_vitals:
        evaluated = classify_direction(
            before_vitals.get(vital),
            final_vitals.get(vital),
            vital,
        )
        effective_directions[vital] = (
            {"up": "increase", "down": "decrease", "stable": "stable"}.get(evaluated)
            if evaluated is not None
            else None
        )
        rule_value = rule_prediction_vitals.get(vital)
        final_value = final_vitals.get(vital)
        residuals[vital] = (
            float(final_value) - float(rule_value)
            if _is_number(final_value) and _is_number(rule_value)
            else None
        )

    return (
        final_vitals,
        resolved_magnitudes,
        magnitude_sources,
        effective_directions,
        residuals,
    )


def _resolve_directions(
    *,
    requested_vitals: list[str],
    before_vitals: dict[str, Any],
    rule_vitals: dict[str, Any],
    llm_directions: dict[str, str | None],
) -> tuple[dict[str, str], dict[str, str]]:
    selected: dict[str, str] = {}
    sources: dict[str, str] = {}
    for vital in requested_vitals:
        llm_direction = llm_directions.get(vital)
        if llm_direction in DIRECTION_LABELS:
            selected[vital] = llm_direction
            sources[vital] = "llm"
            continue

        fallback = classify_direction(
            before_vitals.get(vital),
            rule_vitals.get(vital),
            vital,
        )
        selected[vital] = {
            "up": "increase",
            "down": "decrease",
            "stable": "stable",
            None: "stable",
        }[fallback]
        sources[vital] = "rule_fallback"
    return selected, sources


def _call_client(client: LLMClient, prompt: str) -> dict[str, Any]:
    try:
        text = _response_text(client.complete(prompt))
        error = None
    except Exception as exc:
        text = ""
        error = f"{type(exc).__name__}: {exc}"
    api_error = _client_attr(client, "last_api_error")
    if api_error is None and error is not None:
        api_error = error
    return {"text": text, "error": error, "api_error": api_error}


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str):
        return None, "LLM response was not text"
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "No valid JSON object found in LLM response"


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.group(1).strip() for match in _CODE_FENCE_RE.finditer(text))
    first_object = _first_json_object(stripped)
    if first_object is not None:
        candidates.append(first_object)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _first_json_object(text: str) -> str | None:
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def _normalize_direction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "increase": "increase",
        "increased": "increase",
        "up": "increase",
        "rise": "increase",
        "decrease": "decrease",
        "decreased": "decrease",
        "down": "decrease",
        "fall": "decrease",
        "stable": "stable",
        "unchanged": "stable",
        "no_change": "stable",
    }
    return aliases.get(normalized)


def _extract_reasoning(parsed: dict[str, Any]) -> dict[str, Any]:
    raw = parsed.get("reasoning", parsed.get("rationale"))
    if isinstance(raw, str):
        return {"overall": raw}
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if (key in CANONICAL_VITAL_KEYS or key == "overall")
        and (isinstance(value, str) or value is None)
    }


def _case_block(engine_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": engine_input.get("case_id"),
        "category": engine_input.get("category"),
        "scenario_description": engine_input.get("scenario_description"),
        "case_context": _scrub(
            sanitize_case_context(_mapping(engine_input.get("case_context")))
        ),
    }


def _action_block(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_text": action.get("raw_text"),
        "kind_hint": action.get("kind_hint"),
        "params": _scrub(deepcopy(action.get("params", {}))),
    }


def _render_direction_examples(examples: list[dict[str, Any]] | None) -> str:
    if not examples:
        return ""
    rendered = []
    for example in examples:
        rendered.append(
            {
                "case": example.get("case"),
                "before_vitals": example.get("before_vitals"),
                "before_features": example.get("before_features"),
                "action": example.get("action"),
                "expected_output": example.get("stage_1_expected_output"),
            }
        )
    return (
        "Static demonstrations from other cases:\n"
        f"{_json(rendered)}\n\n"
    )


def _render_magnitude_examples(examples: list[dict[str, Any]] | None) -> str:
    if not examples:
        return ""
    rendered = []
    for example in examples:
        rendered.append(
            {
                "case": example.get("case"),
                "before_vitals": example.get("before_vitals"),
                "before_features": example.get("before_features"),
                "action": example.get("action"),
                "fixed_directions": _mapping(
                    example.get("stage_1_expected_output")
                ).get("directions"),
                "rule_based_prediction": example.get("rule_based_prediction"),
                "expected_output": example.get("stage_2_expected_output"),
            }
        )
    return (
        "Static demonstrations from other cases:\n"
        f"{_json(rendered)}\n\n"
    )


def _example_references(
    examples: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not examples:
        return []
    return [
        {
            "case_id": example.get("_case_id"),
            "pair_id": example.get("_pair_id"),
            "role": example.get("_role"),
        }
        for example in examples
        if isinstance(example, dict)
    ]


def _requested_vitals(vital_names: Iterable[str] | None) -> list[str]:
    if vital_names is None:
        return list(CANONICAL_VITAL_KEYS)
    if isinstance(vital_names, str):
        vital_names = [vital_names]
    return [vital for vital in vital_names if vital in CANONICAL_VITAL_KEYS]


def _before_vitals(engine_input: dict[str, Any]) -> dict[str, Any]:
    before = _mapping(engine_input.get("before"))
    return _canonical_vitals(before.get("vitals", {}))


def _canonical_vitals(vitals: Any) -> dict[str, Any]:
    vitals = vitals if isinstance(vitals, dict) else {}
    return {vital: deepcopy(vitals.get(vital)) for vital in CANONICAL_VITAL_KEYS}


def _null_template(vital_names: Iterable[str]) -> dict[str, None]:
    return {vital: None for vital in vital_names if vital in CANONICAL_VITAL_KEYS}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _client_attr(client: Any, name: str) -> Any:
    value = getattr(client, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")
    for attr in ("text", "content"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    return str(response)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub(nested)
            for key, nested in value.items()
            if key not in _FORBIDDEN_PROMPT_KEYS
        }
    if isinstance(value, list):
        return [_scrub(nested) for nested in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "DIRECTION_LABELS",
    "DirectionFirstHybridEngine",
    "LLMClient",
    "MIN_DIRECTIONAL_MAGNITUDES",
    "STAGE1_PROMPT_PROFILES",
    "build_calibrated_direction_prompt",
    "build_calibrated_v3_direction_prompt",
    "build_calibrated_v4_direction_prompt",
    "build_direction_prompt",
    "build_magnitude_prompt",
    "compose_directional_vitals",
    "parse_llm_directions",
    "parse_llm_magnitudes",
]
