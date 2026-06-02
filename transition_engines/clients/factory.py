"""Model registry and factory mapping a model id to its provider adapter."""
from __future__ import annotations

from typing import Any

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIChatClient, OpenAIResponsesClient


# temperature: "tunable" passes the requested temperature through; "fixed" omits
# it (reasoning models that reject an explicit temperature).
# reasoning models (gpt-5 family) spend the output-token budget on internal
# reasoning, so they need a higher floor (min_output_tokens) and an explicit
# (low) reasoning effort, or the JSON answer comes back empty/truncated.
_REASONING_TOKEN_FLOOR = 4000
_REASONING_EFFORT = "low"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {"client": OpenAIChatClient, "temperature": "tunable"},
    "gpt-4o": {"client": OpenAIChatClient, "temperature": "tunable"},
    "gpt-4.1": {"client": OpenAIChatClient, "temperature": "tunable"},
    "gpt-5": {
        "client": OpenAIResponsesClient,
        "temperature": "fixed",
        "reasoning": True,
    },
    "gpt-5.5": {
        "client": OpenAIResponsesClient,
        "temperature": "tunable",
        "reasoning": True,
    },
    "gemini-2.5-flash": {"client": GeminiClient, "temperature": "tunable"},
    "claude-sonnet-4-6": {"client": AnthropicClient, "temperature": "tunable"},
}


def build_llm_client(
    model_id: str,
    *,
    response_schema_type: str,
    temperature: float = 0.0,
    max_output_tokens: int = 300,
    api_key: str | None = None,
) -> BaseLLMClient:
    """Construct the provider adapter for ``model_id``.

    Raises ValueError for an unknown model id. Never raises on a missing API
    key; the returned client degrades to empty JSON on ``complete()``.
    """

    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_id: {model_id!r}. Known models: {known}")

    resolved_temperature = (
        None if entry["temperature"] == "fixed" else temperature
    )
    is_reasoning = entry.get("reasoning", False)
    resolved_max_output_tokens = (
        max(max_output_tokens, _REASONING_TOKEN_FLOOR)
        if is_reasoning
        else max_output_tokens
    )
    reasoning_effort = _REASONING_EFFORT if is_reasoning else None

    client_cls = entry["client"]
    return client_cls(
        model=model_id,
        response_schema_type=response_schema_type,
        temperature=resolved_temperature,
        max_output_tokens=resolved_max_output_tokens,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )


__all__ = ["MODEL_REGISTRY", "build_llm_client"]
