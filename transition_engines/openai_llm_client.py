"""OpenAI SDK client adapter for transition-pair LLM engines."""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .common import CANONICAL_VITAL_KEYS


MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set. Please export it before using --llm-mode real."
)
EMPTY_ADJUSTMENTS_RESPONSE = json.dumps({"adjustments": {}}, sort_keys=True)
EMPTY_VITALS_RESPONSE = json.dumps({"vitals": {}}, sort_keys=True)
RESPONSE_SCHEMA_TYPES = {"adjustments", "vitals"}


class OpenAILLMClient:
    """OpenAI Responses API wrapper exposing complete(prompt) for engines."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        temperature: float = 0.0,
        max_output_tokens: int = 300,
        response_schema_type: str = "adjustments",
        client: Any | None = None,
    ) -> None:
        if response_schema_type not in RESPONSE_SCHEMA_TYPES:
            raise ValueError(
                f"Unsupported response_schema_type: {response_schema_type!r}"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(MISSING_API_KEY_MESSAGE)

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_schema_type = response_schema_type
        self.last_api_error: str | None = None
        self._api_key = api_key
        self._client = client if client is not None else OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> str:
        """Return model text, or empty schema-compatible JSON if the API call fails."""

        self.last_api_error = None
        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                text={"format": _response_json_schema(self.response_schema_type)},
            )
        except Exception as exc:
            self.last_api_error = _sanitize_api_error(exc, self._api_key)
            return _empty_response(self.response_schema_type)

        return _response_text(response)


def _response_json_schema(response_schema_type: str) -> dict[str, Any]:
    if response_schema_type == "adjustments":
        return _adjustments_json_schema()
    if response_schema_type == "vitals":
        return _vitals_json_schema()
    raise ValueError(f"Unsupported response_schema_type: {response_schema_type!r}")


def _adjustments_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "hybrid_vital_adjustments",
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "adjustments": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        vital: {"type": ["number", "null"]}
                        for vital in CANONICAL_VITAL_KEYS
                    },
                },
                "reasoning": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        vital: {"type": ["string", "null"]}
                        for vital in CANONICAL_VITAL_KEYS
                    },
                }
            },
            "required": ["adjustments"],
        },
    }


def _vitals_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "pure_llm_vital_predictions",
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "vitals": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        vital: {"type": ["number", "null"]}
                        for vital in CANONICAL_VITAL_KEYS
                    },
                    "required": list(CANONICAL_VITAL_KEYS),
                },
                "reasoning": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        vital: {"type": ["string", "null"]}
                        for vital in CANONICAL_VITAL_KEYS
                    },
                }
            },
            "required": ["vitals"],
        },
    }


def _empty_response(response_schema_type: str) -> str:
    if response_schema_type == "adjustments":
        return EMPTY_ADJUSTMENTS_RESPONSE
    if response_schema_type == "vitals":
        return EMPTY_VITALS_RESPONSE
    raise ValueError(f"Unsupported response_schema_type: {response_schema_type!r}")


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    chunks: list[str] = []
    for output_item in _as_list(getattr(response, "output", None)):
        for content_item in _as_list(getattr(output_item, "content", None)):
            content_text = getattr(content_item, "text", None)
            if isinstance(content_text, str):
                chunks.append(content_text)
                continue
            content_text = _mapping_get(content_item, "text")
            if isinstance(content_text, str):
                chunks.append(content_text)

    if chunks:
        return "".join(chunks)

    return str(response)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _redact_api_key(text: str, api_key: str) -> str:
    if api_key and api_key in text:
        return text.replace(api_key, "[REDACTED_OPENAI_API_KEY]")
    return text


def _sanitize_api_error(exc: Exception, api_key: str) -> str:
    message = _redact_api_key(str(exc), api_key)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


__all__ = [
    "EMPTY_ADJUSTMENTS_RESPONSE",
    "EMPTY_VITALS_RESPONSE",
    "MISSING_API_KEY_MESSAGE",
    "OpenAILLMClient",
]
