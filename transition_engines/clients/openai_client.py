"""OpenAI adapters: Responses API (gpt-5 / gpt-5.5) and Chat Completions (4o family)."""
from __future__ import annotations

from typing import Any

from .base import BaseLLMClient


class OpenAIResponsesClient(BaseLLMClient):
    """Responses API client for newer models, mirroring the production client."""

    env_vars = ("OPENAI_API_KEY",)

    def _call(self, prompt: str) -> Any:
        from openai import OpenAI

        from ..openai_llm_client import _response_json_schema

        client = OpenAI(api_key=self._api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "text": {"format": _response_json_schema(self.response_schema_type)},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return client.responses.create(**kwargs)


class OpenAIChatClient(BaseLLMClient):
    """Chat Completions client for gpt-4o / gpt-4o-mini / gpt-4.1.

    Uses JSON mode (``response_format={"type": "json_object"}``) rather than
    strict json_schema; the engines' tolerant JSON parser handles extraction.
    """

    env_vars = ("OPENAI_API_KEY",)

    def _call(self, prompt: str) -> Any:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


__all__ = ["OpenAIChatClient", "OpenAIResponsesClient"]
