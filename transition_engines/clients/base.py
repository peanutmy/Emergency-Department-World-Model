"""Shared infrastructure for multi-provider LLM client adapters.

Every adapter exposes ``complete(prompt) -> str`` (the ``LLMClient`` Protocol the
engines depend on) plus ``model`` and ``last_api_error`` attributes consumed by
engine metadata. SDK imports are lazy (inside ``_call``), so importing this
package never requires a provider SDK to be installed, and a missing API key
degrades gracefully to schema-compatible empty JSON instead of raising.
"""
from __future__ import annotations

import json
import os
from typing import Any


EMPTY_RESPONSES = {
    "adjustments": json.dumps({"adjustments": {}}, sort_keys=True),
    "vitals": json.dumps({"vitals": {}}, sort_keys=True),
}


def empty_response(response_schema_type: str) -> str:
    try:
        return EMPTY_RESPONSES[response_schema_type]
    except KeyError:
        raise ValueError(
            f"Unsupported response_schema_type: {response_schema_type!r}"
        ) from None


def redact_secret(text: str, secret: str | None) -> str:
    if secret and secret in text:
        return text.replace(secret, "[REDACTED_API_KEY]")
    return text


def sanitize_error(exc: Exception, secret: str | None) -> str:
    message = redact_secret(str(exc), secret)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def response_text(response: Any) -> str:
    """Coerce a provider response object into plain text, best effort."""

    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    for attr in ("text", "content"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    return str(response)


class BaseLLMClient:
    """Common key-resolution + graceful-degradation skeleton for adapters."""

    #: Environment variables checked in order when ``api_key`` is not given.
    env_vars: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        model: str,
        response_schema_type: str,
        temperature: float | None = 0.0,
        max_output_tokens: int = 300,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if response_schema_type not in EMPTY_RESPONSES:
            raise ValueError(
                f"Unsupported response_schema_type: {response_schema_type!r}"
            )
        self.model = model
        self.response_schema_type = response_schema_type
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Set for reasoning models (gpt-5 family); None for non-reasoning models.
        self.reasoning_effort = reasoning_effort
        # An explicitly supplied api_key (even "") is used verbatim; only None
        # falls back to the environment. This keeps tests/degrade deterministic.
        self._api_key = api_key if api_key is not None else self._key_from_env()
        self.last_api_error: str | None = None

    def _key_from_env(self) -> str | None:
        for name in self.env_vars:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def complete(self, prompt: str) -> str:
        self.last_api_error = None
        if not self._api_key:
            self.last_api_error = f"{self.env_vars[0]} is not set"
            return empty_response(self.response_schema_type)
        try:
            return response_text(self._call(prompt))
        except Exception as exc:  # provider/network/SDK failures all degrade
            self.last_api_error = sanitize_error(exc, self._api_key)
            return empty_response(self.response_schema_type)

    def _call(self, prompt: str) -> Any:
        raise NotImplementedError


__all__ = [
    "BaseLLMClient",
    "EMPTY_RESPONSES",
    "empty_response",
    "redact_secret",
    "response_text",
    "sanitize_error",
]
