"""Hybrid LLM residual engine for educational ED transition pairs."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Protocol

from .common import CANONICAL_VITAL_KEYS, EngineOutput, sanitize_case_context
from .rule_based import RuleBasedEngine, clamp_vitals, predict_features


class LLMClient(Protocol):
    """Minimal LLM dependency surface used by HybridEngine."""

    def complete(self, prompt: str) -> str:
        """Return the model response text for a prompt."""


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FORBIDDEN_PROMPT_KEYS = {
    "label",
    "modifier_text",
    "source",
    "supporting_findings",
    "target",
}


class HybridEngine:
    """Apply LLM vital residuals on top of rule-based vitals and features."""

    def __init__(self, rule_engine: RuleBasedEngine, llm_client: LLMClient) -> None:
        self.rule_engine = rule_engine
        self.llm_client = llm_client

    def predict(
        self,
        engine_input: dict[str, Any],
        target_vital_names: Iterable[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> EngineOutput:
        """Predict post-action vitals and deterministic non-vital features.

        target_vital_names=None allows adjustments for all canonical vitals.
        target_vital_names=[] allows no adjustments.
        examples=None (the runtime default) produces the zero-shot prompt; a
        non-empty list injects few-shot demonstrations ahead of the live case.
        """

        rule_output = self.rule_engine.predict(engine_input)
        rule_vitals = _canonical_vitals(
            rule_output.get("prediction", {}).get("vitals", {})
        )
        rule_features = _prediction_features(rule_output, engine_input)
        requested_vitals = _requested_vitals(target_vital_names)

        prompt = build_llm_prompt(
            engine_input=engine_input,
            rule_prediction_vitals=rule_vitals,
            target_vital_names=requested_vitals,
            examples=examples,
        )
        try:
            llm_text = _complete(self.llm_client, prompt)
            llm_error = None
        except Exception as exc:
            llm_text = ""
            llm_error = repr(exc)
        llm_api_error = _client_attr(self.llm_client, "last_api_error")
        if llm_api_error is None and llm_error is not None:
            llm_api_error = llm_error
        if llm_error is None:
            (
                llm_adjustments,
                llm_reasoning,
                llm_parse_error,
            ) = _parse_llm_adjustments_with_error(llm_text)
        else:
            llm_adjustments = {}
            llm_reasoning = {}
            llm_parse_error = None

        final_vitals = merge_vitals(
            rule_prediction_vitals=rule_vitals,
            before_vitals=_before_vitals(engine_input),
            adjustments=llm_adjustments,
            target_vital_names=(
                requested_vitals if target_vital_names is not None else None
            ),
        )

        return {
            "prediction": {"vitals": final_vitals, "features": rule_features},
            "metadata": {
                "engine": "hybrid",
                "rule_prediction": {
                    "vitals": rule_vitals,
                    "features": rule_features,
                },
                "llm_adjustments": llm_adjustments,
                "llm_reasoning": llm_reasoning,
                "llm_error": llm_error,
                "llm_raw_response": llm_text,
                "llm_parse_error": llm_parse_error,
                "llm_api_error": llm_api_error,
                "model": _client_attr(self.llm_client, "model"),
            },
        }


def build_llm_prompt(
    *,
    engine_input: dict[str, Any],
    rule_prediction_vitals: dict[str, Any],
    target_vital_names: list[str],
    examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build a leak-resistant prompt from explicitly allowed input fields.

    examples=None preserves the exact zero-shot prompt used at runtime. A
    non-empty list is rendered as a few-shot block immediately before the
    live ``Case:`` section.
    """

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    case_context = _scrub_forbidden_keys(
        sanitize_case_context(_mapping(engine_input.get("case_context")))
    )
    case_context_block = {
        "case_id": engine_input.get("case_id"),
        "category": engine_input.get("category"),
        "scenario_description": engine_input.get("scenario_description"),
        "case_context": case_context,
    }

    return f"""You are a transition prediction correction module for an educational emergency medicine simulation.

You are given:
1. Case-level context explicitly provided by the scenario.
2. The pre-action patient state.
3. The structured action.
4. A deterministic rule-based prediction.

Your task:
Return numeric residual adjustments to the rule-based prediction for the requested vital names only.

The rule-based prediction is only a rough baseline. It may be wrong.
Do not assume the rule-based prediction is correct.
Your job is to decide whether the rule-based prediction is plausible given the provided case context, pre-action state, and action.
If it is implausible, return a correction adjustment.
If it is plausible, return null or 0 for that vital.

Important rules:
- Use only the provided information.
- Do not infer hidden diagnoses beyond the provided scenario text.
- Do not use any post-action values.
- Output JSON only.
- Adjust only these requested vitals: {_json(target_vital_names)}
- Do not change vitals outside this requested list.
- Adjustments may be small, zero, or large depending on whether the rule-based prediction is consistent with the provided scenario and action.
- Keep all final vitals physiologically plausible.
- Include concise rationale notes for requested vitals only. Use at most one sentence per vital and do not include step-by-step deliberation.
- Each rationale must explain the numeric residual: mention the rule-based value, adjustment amount, final value, and clinical reason.

Action-specific guidance:
- For no_action:
  Treat this as natural time progression, delayed care, or lack of intervention.
  Do not default to no change when the patient is unstable.
  Use the current abnormalities and explicitly provided scenario context to judge whether the requested vitals should worsen, improve, or remain stable.

- For airway_management:
  Pay close attention to procedure, stage, intubated, bvm_before, before oxygenation, before respiratory rate, and raw_text.
  Completed successful intubation may improve oxygenation and produce a ventilated respiratory pattern.
  Missing BVM/preoxygenation, peri-intubation difficulty, incomplete airway management, or delayed airway management may worsen O2Sat, BP, HR, or RR if supported by the provided context.
  Do not blindly assume every airway_management action improves all requested vitals.

- For oxygen_support:
  Use oxygen_device, FiO2, PEEP_used, and PEEP_cmH2O.
  Higher oxygen support or PEEP may improve O2Sat, but the patient may remain hypoxemic if the case context indicates severe respiratory distress.

- For medication-like actions:
  Use kind_hint, raw_text, drug_name, dose, and unit if provided.
  Do not assume every medication causes immediate improvement.
  Some medication effects may be delayed, limited, or harmful depending only on the provided scenario context.

- For antidote or toxicology-related actions:
  Use the explicitly provided scenario context and action text.
  Severe toxicity may continue to worsen with no_action or may improve only partially after treatment.

- For anticoagulation:
  Do not assume immediate improvement.
  If the provided scenario context suggests anticoagulation is inappropriate or harmful, adjust accordingly based only on the provided context.

{_render_examples_block(examples)}Case:
{_json(case_context_block)}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub_forbidden_keys(deepcopy(before.get("features", {}))))}

Action:
raw_text: {_json(action.get("raw_text"))}
kind_hint: {_json(action.get("kind_hint"))}
params: {_json(_scrub_forbidden_keys(deepcopy(action.get("params", {}))))}

Rule-based prediction:
{_json(_canonical_vitals(rule_prediction_vitals))}

Return JSON only:
{_json({"adjustments": _adjustment_template(target_vital_names), "reasoning": _reasoning_template(target_vital_names)})}"""


def parse_llm_adjustments(text: str) -> dict[str, Any]:
    """Parse canonical numeric residuals from LLM JSON text.

    Markdown code fences are accepted. Invalid JSON returns an empty dict. Missing
    canonical fields in otherwise valid JSON are represented as null/None.
    """

    adjustments, _, _ = _parse_llm_adjustments_with_error(text)
    return adjustments


def _parse_llm_adjustments_with_error(
    text: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    if not isinstance(text, str):
        return {}, {}, "LLM response was not text"

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return _extract_adjustments(parsed), _extract_reasoning(parsed), None
    return {}, {}, "No valid JSON object found in LLM response"


def merge_vitals(
    *,
    rule_prediction_vitals: dict[str, Any],
    before_vitals: dict[str, Any],
    adjustments: dict[str, Any],
    target_vital_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Merge numeric LLM residuals into rule-predicted canonical vitals."""

    final_vitals = _canonical_vitals(rule_prediction_vitals)
    allowed_vitals = (
        set(CANONICAL_VITAL_KEYS)
        if target_vital_names is None
        else set(_requested_vitals(target_vital_names))
    )

    for vital in CANONICAL_VITAL_KEYS:
        if vital not in allowed_vitals:
            continue
        adjustment = adjustments.get(vital)
        if not _is_number(adjustment):
            continue

        rule_value = final_vitals.get(vital)
        if _is_number(rule_value):
            final_vitals[vital] = rule_value + adjustment
            continue

        before_value = before_vitals.get(vital)
        if _is_number(before_value):
            final_vitals[vital] = before_value + adjustment

    return clamp_vitals(final_vitals)


def _complete(llm_client: LLMClient, prompt: str) -> str:
    complete = getattr(llm_client, "complete", None)
    if callable(complete):
        return _response_text(complete(prompt))
    if callable(llm_client):
        return _response_text(llm_client(prompt))
    raise TypeError("llm_client must expose complete(prompt) or be callable")


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


def _extract_adjustments(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {vital: None for vital in CANONICAL_VITAL_KEYS}
    raw_adjustments = parsed.get("adjustments", {})
    if not isinstance(raw_adjustments, dict):
        raw_adjustments = {}

    adjustments: dict[str, Any] = {}
    for vital in CANONICAL_VITAL_KEYS:
        value = raw_adjustments.get(vital)
        adjustments[vital] = value if _is_number(value) else None
    return adjustments


def _extract_reasoning(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}

    raw_reasoning = parsed.get("reasoning")
    if raw_reasoning is None:
        raw_reasoning = parsed.get("rationale")

    if isinstance(raw_reasoning, str):
        return {"overall": raw_reasoning}
    if not isinstance(raw_reasoning, dict):
        return {}

    reasoning: dict[str, Any] = {}
    for vital in CANONICAL_VITAL_KEYS:
        value = raw_reasoning.get(vital)
        if isinstance(value, str) or value is None and vital in raw_reasoning:
            reasoning[vital] = value

    overall = raw_reasoning.get("overall")
    if isinstance(overall, str):
        reasoning["overall"] = overall
    return reasoning


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.group(1).strip() for match in _CODE_FENCE_RE.finditer(text))

    json_object = _first_json_object(stripped)
    if json_object is not None:
        candidates.append(json_object)

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


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


def _canonical_vitals(vitals: Any) -> dict[str, Any]:
    if not isinstance(vitals, dict):
        vitals = {}
    return {vital: deepcopy(vitals.get(vital)) for vital in CANONICAL_VITAL_KEYS}


def _before_vitals(engine_input: dict[str, Any]) -> dict[str, Any]:
    return _canonical_vitals(_mapping(engine_input.get("before")).get("vitals", {}))


def _prediction_features(
    rule_output: dict[str, Any],
    engine_input: dict[str, Any],
) -> dict[str, Any]:
    prediction = _mapping(rule_output.get("prediction"))
    features = prediction.get("features")
    if isinstance(features, dict):
        return deepcopy(features)
    return predict_features(engine_input)


def _requested_vitals(vital_names: Iterable[str] | None) -> list[str]:
    if vital_names is None:
        return list(CANONICAL_VITAL_KEYS)
    if isinstance(vital_names, str):
        vital_names = [vital_names]
    return [vital for vital in vital_names if vital in CANONICAL_VITAL_KEYS]


def _adjustment_template(vital_names: Iterable[str]) -> dict[str, None]:
    return {vital: None for vital in vital_names if vital in CANONICAL_VITAL_KEYS}


def _reasoning_template(vital_names: Iterable[str]) -> dict[str, None]:
    return {vital: None for vital in vital_names if vital in CANONICAL_VITAL_KEYS}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _client_attr(llm_client: Any, name: str) -> Any:
    value = getattr(llm_client, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _render_examples_block(examples: list[dict[str, Any]] | None) -> str:
    """Render few-shot demonstrations, or '' for the zero-shot default."""

    if not examples:
        return ""
    public = [
        {
            key: value
            for key, value in example.items()
            if not str(key).startswith("_")
        }
        if isinstance(example, dict)
        else example
        for example in examples
    ]
    return (
        "Few-shot examples (illustrative, drawn from other cases; each shows "
        "the expected output for that input):\n"
        f"{_json(public)}\n\n"
    )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _scrub_forbidden_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_forbidden_keys(nested)
            for key, nested in value.items()
            if key not in _FORBIDDEN_PROMPT_KEYS
        }
    if isinstance(value, list):
        return [_scrub_forbidden_keys(nested) for nested in value]
    return value


__all__ = [
    "HybridEngine",
    "LLMClient",
    "build_llm_prompt",
    "merge_vitals",
    "parse_llm_adjustments",
]
