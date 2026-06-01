"""LLM client wrappers for v1.3.1 runtime agents."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from typing import Any, Protocol

try:  # pragma: no cover - exercised through monkeypatched clients in tests.
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


MISSING_OPENAI_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set. Export it before using --agent-mode real "
    "or --physiology-mode hybrid or --fact-extractor llm."
)
DEFAULT_AGENT_RESPONSE = {"verbal_action": None, "action": None}
DEFAULT_VERBAL_ONLY_RESPONSE = {"verbal_action": None}
DEFAULT_CLINICIAN_DECISION_RESPONSE = {
    "action": None,
    "verbal_decision": None,
}
DEFAULT_PATIENT_DECISION_RESPONSE = {
    "speaker": "patient",
    "should_speak": False,
    "target": None,
    "intent": "stay silent",
    "reasoning_summary": "No patient response is needed from the observation.",
    "key_points": [],
    "forbidden_points": [],
    "requires_response": False,
}


class LLMClient(Protocol):
    """Minimal runtime LLM surface for agent generation."""

    def generate(self, prompt: str) -> str:
        """Return final model text for a prompt."""


class FakeLLMClient:
    """Deterministic scripted LLM client for fake mode and tests."""

    provider = "fake"
    model = "fake-scripted"
    last_api_error: str | None = None

    def __init__(
        self,
        responses: Iterable[str | Mapping[str, Any]] | None = None,
        *,
        default_response: str | Mapping[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._default_response = (
            DEFAULT_AGENT_RESPONSE
            if default_response is None
            else default_response
        )
        self.prompts: list[str] = []
        self.prompt_lengths: list[int] = []
        self._index = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.prompt_lengths.append(len(prompt))
        if self._index >= len(self._responses):
            return _response_to_text(self._default_response)
        response = self._responses[self._index]
        self._index += 1
        return _response_to_text(response)

    def complete(self, prompt: str) -> str:
        """Compatibility alias for existing transition-engine clients."""

        return self.generate(prompt)

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


class OpenAICompatibleLLMClient:
    """OpenAI-compatible Responses API client for explicit real mode."""

    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 700,
        client: Any | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(MISSING_OPENAI_API_KEY_MESSAGE)
        if client is None and OpenAI is None:
            raise RuntimeError(
                "The openai package is required for OpenAI-compatible real mode."
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.last_api_error: str | None = None
        self._api_key = api_key
        self._client = client if client is not None else OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        self.last_api_error = None
        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            self.last_api_error = _sanitize_api_error(exc, self._api_key)
            raise RuntimeError(
                f"OpenAI-compatible LLM request failed: {self.last_api_error}"
            ) from None
        return _extract_response_text(response)

    def complete(self, prompt: str) -> str:
        """Compatibility alias for existing transition-engine clients."""

        return self.generate(prompt)

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


RealLLMClient = OpenAICompatibleLLMClient


def _response_to_text(response: str | Mapping[str, Any]) -> str:
    if isinstance(response, str):
        return response
    return json.dumps(response)


def _extract_response_text(response: Any) -> str:
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
            if isinstance(content_item, Mapping):
                content_text = content_item.get("text")
                if isinstance(content_text, str):
                    chunks.append(content_text)
    if chunks:
        return "".join(chunks)
    return str(response)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sanitize_api_error(exc: Exception, api_key: str) -> str:
    message = str(exc)
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED_OPENAI_API_KEY]")
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


__all__ = [
    "DEFAULT_AGENT_RESPONSE",
    "DEFAULT_CLINICIAN_DECISION_RESPONSE",
    "DEFAULT_PATIENT_DECISION_RESPONSE",
    "DEFAULT_VERBAL_ONLY_RESPONSE",
    "FakeLLMClient",
    "LLMClient",
    "MISSING_OPENAI_API_KEY_MESSAGE",
    "OpenAICompatibleLLMClient",
    "RealLLMClient",
]
