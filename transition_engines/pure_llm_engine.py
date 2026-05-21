"""Pure LLM engine for educational ED transition pairs."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Protocol

from .common import CANONICAL_VITAL_KEYS, EngineOutput, VitalKey, sanitize_case_context
from .rule_based import predict_features


class LLMClient(Protocol):
    """Minimal LLM dependency surface used by PureLLMEngine."""

    def complete(self, prompt: str) -> str:
        """Return the model response text for a prompt."""


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FORBIDDEN_PROMPT_KEYS = {
    "label",
    "modifier_text",
    "rule_prediction",
    "source",
    "supporting_findings",
    "target",
}
_VITAL_BOUNDS: dict[VitalKey, tuple[float, float]] = {
    "HR": (0, 250),
    "BP_sys": (40, 260),
    "BP_dia": (20, 160),
    "RR": (0, 60),
    "O2Sat": (0, 100),
    "T": (30, 43),
}


class PureLLMEngine:
    """Predict post-action vitals with LLM and features with deterministic rules."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def predict(
        self,
        engine_input: dict[str, Any],
        target_vital_names: Iterable[str] | None = None,
    ) -> EngineOutput:
        """Predict post-action vitals plus rule-based non-vital features."""

        before_vitals = _before_vitals(engine_input)
        requested_vitals = _requested_vitals(target_vital_names)
        prompt = build_llm_prompt(
            engine_input=engine_input,
            target_vital_names=requested_vitals,
        )

        try:
            llm_text = _complete(self.llm_client, prompt)
            llm_call_error = None
        except Exception as exc:
            llm_text = ""
            llm_call_error = f"LLM call failed: {type(exc).__name__}"

        llm_api_error = _client_attr(self.llm_client, "last_api_error")
        if llm_api_error is None and llm_call_error is not None:
            llm_api_error = llm_call_error

        if llm_call_error is None:
            llm_vitals, llm_reasoning, llm_parse_error = (
                _parse_llm_vitals_with_error(llm_text)
            )
        else:
            llm_vitals = _null_vitals()
            llm_reasoning = {}
            llm_parse_error = llm_call_error

        final_vitals = merge_vitals(
            before_vitals=before_vitals,
            llm_vitals=llm_vitals,
            target_vital_names=(
                requested_vitals if target_vital_names is not None else None
            ),
        )
        final_features = predict_features(engine_input)

        return {
            "prediction": {"vitals": final_vitals, "features": final_features},
            "metadata": {
                "engine": "pure_llm",
                "llm_raw_response": llm_text,
                "llm_reasoning": llm_reasoning,
                "llm_parse_error": llm_parse_error,
                "llm_api_error": llm_api_error,
                "model": _client_attr(self.llm_client, "model"),
            },
        }


def build_llm_prompt(
    *,
    engine_input: dict[str, Any],
    target_vital_names: Iterable[str],
) -> str:
    """Build a leak-resistant prompt from explicitly allowed input fields."""

    before = _mapping(engine_input.get("before"))
    action = _mapping(engine_input.get("action"))
    case_context = _scrub_forbidden_keys(
        sanitize_case_context(_mapping(engine_input.get("case_context")))
    )
    case_block = {
        "case_id": engine_input.get("case_id"),
        "category": engine_input.get("category"),
        "scenario_description": engine_input.get("scenario_description"),
        "case_context": case_context,
    }
    requested_vitals = _requested_vitals(target_vital_names)

    return f"""You are predicting post-action vital signs for an educational emergency medicine transition pair.

You are given:
1. Case-level context explicitly provided by the scenario.
2. The pre-action patient state.
3. The structured action.

Task:
Predict post-action values for the requested vital names only.

Important instructions:
- Use only the explicitly provided case context, pre-action state, and action.
- Do not infer hidden diagnoses beyond the provided scenario text.
- Do not use any post-action values.
- Output JSON only.
- Focus on predicting these requested vital names: {_json(requested_vitals)}
- For vitals outside the requested list, either keep them close to before values or output null.
- If a requested vital is not expected to change, predict the same value as before.
- Keep all vitals physiologically plausible.
- Include concise rationale notes for requested vitals only. Use at most one sentence per vital and do not include step-by-step deliberation.
- Each rationale must explain the numeric prediction: mention the before value, predicted final value, direction or magnitude of change, and clinical reason.

Action-specific guidance:
- For no_action:
  Treat this as natural time progression, delayed care, or lack of intervention.
  Do not default to no change when the patient is unstable.
  Use current abnormalities and explicitly provided scenario context to judge whether requested vitals worsen, improve, or remain stable.

- For airway_management:
  Pay attention to procedure, stage, intubated, bvm_before, before oxygenation, before RR, and raw_text.
  Completed successful intubation may improve oxygenation and produce a ventilated respiratory pattern.
  Missing BVM/preoxygenation, peri-intubation difficulty, incomplete airway management, or delayed airway management may worsen O2Sat, BP, HR, or RR if supported by provided context.
  Do not blindly assume all airway_management actions improve all vitals.

- For oxygen_support:
  Use oxygen_device, FiO2, PEEP_used, and PEEP_cmH2O.
  Higher oxygen support or PEEP may improve O2Sat, but the patient may remain hypoxemic if the case context indicates severe respiratory distress.

- For medication-like actions:
  Use kind_hint, raw_text, drug_name, dose, and unit if provided.
  Do not assume every medication causes immediate improvement.
  Some medication effects may be delayed, limited, or harmful depending only on provided scenario context.

- For antidote or toxicology-related actions:
  Use explicitly provided scenario context and action text.
  Severe toxicity may continue to worsen with no_action or may improve only partially after treatment.

- For anticoagulation:
  Do not assume immediate improvement.
  If provided scenario context suggests anticoagulation is inappropriate or harmful, predict accordingly based only on provided context.

Case:
{_json(case_block)}

Before vitals:
{_json(_canonical_vitals(before.get("vitals", {})))}

Before features:
{_json(_scrub_forbidden_keys(deepcopy(before.get("features", {}))))}

Action:
raw_text: {_json(action.get("raw_text"))}
kind_hint: {_json(action.get("kind_hint"))}
params: {_json(_scrub_forbidden_keys(deepcopy(action.get("params", {}))))}

Return JSON only:
{_json({"vitals": _vital_template(), "reasoning": _reasoning_template(requested_vitals)})}"""


def parse_llm_vitals(text: str) -> dict[str, Any]:
    """Parse canonical predicted vitals from LLM JSON text.

    Markdown code fences are accepted. Invalid JSON returns all canonical fields
    as null/None. Missing canonical fields in otherwise valid JSON are also
    represented as null/None.
    """

    vitals, _, _ = _parse_llm_vitals_with_error(text)
    return vitals


def merge_vitals(
    *,
    before_vitals: dict[str, Any],
    llm_vitals: dict[str, Any],
    target_vital_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Apply numeric LLM vital predictions to copied pre-action vitals."""

    final_vitals = _canonical_vitals(before_vitals)
    allowed_vitals = (
        set(CANONICAL_VITAL_KEYS)
        if target_vital_names is None
        else set(_requested_vitals(target_vital_names))
    )

    for vital in CANONICAL_VITAL_KEYS:
        if vital not in allowed_vitals:
            continue
        prediction = llm_vitals.get(vital)
        if _is_number(prediction):
            final_vitals[vital] = prediction

    return clamp_vitals(final_vitals)


def clamp_vitals(vitals: dict[str, Any]) -> dict[str, Any]:
    """Clamp present canonical vitals to physiologic bounds."""

    for vital, (lower, upper) in _VITAL_BOUNDS.items():
        value = vitals.get(vital)
        if _is_number(value):
            vitals[vital] = min(upper, max(lower, value))
    return vitals


def _parse_llm_vitals_with_error(
    text: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    if not isinstance(text, str):
        return _null_vitals(), {}, "LLM response was not text"

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        vitals, error = _extract_vitals(parsed)
        return vitals, _extract_reasoning(parsed), error
    return _null_vitals(), {}, "No valid JSON object found in LLM response"


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


def _extract_vitals(parsed: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(parsed, dict):
        return _null_vitals(), "LLM JSON response was not an object"

    if "vitals" in parsed:
        raw_vitals = parsed.get("vitals")
        if not isinstance(raw_vitals, dict):
            return _null_vitals(), "LLM JSON field 'vitals' was not an object"
        return _extract_numeric_vitals(raw_vitals), None

    if any(vital in parsed for vital in CANONICAL_VITAL_KEYS):
        return _extract_numeric_vitals(parsed), None

    return (
        _null_vitals(),
        "LLM JSON response did not contain 'vitals' or direct canonical vitals",
    )


def _extract_numeric_vitals(raw_vitals: dict[str, Any]) -> dict[str, Any]:
    return {
        vital: raw_vitals.get(vital) if _is_number(raw_vitals.get(vital)) else None
        for vital in CANONICAL_VITAL_KEYS
    }


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


def _before_vitals(engine_input: dict[str, Any]) -> dict[str, Any]:
    return _canonical_vitals(_mapping(engine_input.get("before")).get("vitals", {}))


def _canonical_vitals(vitals: Any) -> dict[str, Any]:
    if not isinstance(vitals, dict):
        vitals = {}
    return {vital: deepcopy(vitals.get(vital)) for vital in CANONICAL_VITAL_KEYS}


def _requested_vitals(vital_names: Iterable[str] | None) -> list[str]:
    if vital_names is None:
        return list(CANONICAL_VITAL_KEYS)
    if isinstance(vital_names, str):
        vital_names = [vital_names]
    return [vital for vital in vital_names if vital in CANONICAL_VITAL_KEYS]


def _vital_template() -> dict[str, None]:
    return {vital: None for vital in CANONICAL_VITAL_KEYS}


def _reasoning_template(vital_names: Iterable[str]) -> dict[str, None]:
    return {vital: None for vital in vital_names if vital in CANONICAL_VITAL_KEYS}


def _null_vitals() -> dict[str, None]:
    return {vital: None for vital in CANONICAL_VITAL_KEYS}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _client_attr(llm_client: Any, name: str) -> Any:
    value = getattr(llm_client, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


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
    "LLMClient",
    "PureLLMEngine",
    "build_llm_prompt",
    "clamp_vitals",
    "merge_vitals",
    "parse_llm_vitals",
]
