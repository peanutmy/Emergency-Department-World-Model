"""Tests for the multi-provider LLM client factory and graceful degradation.

These run without any API keys or provider SDKs installed: the degrade path
never imports an SDK, so it exercises routing + missing-key behavior only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from transition_engines.clients import MODEL_REGISTRY, build_llm_client
from transition_engines.clients.openai_client import (
    OpenAIChatClient,
    OpenAIResponsesClient,
)
from transition_engines.clients.gemini_client import GeminiClient
from transition_engines.clients.anthropic_client import AnthropicClient


def test_registry_has_expected_models_and_excludes_4_turbo() -> None:
    assert set(MODEL_REGISTRY) == {
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gpt-5.5",
        "gemini-2.5-flash",
        "claude-sonnet-4-6",
    }
    assert "gpt-4-turbo" not in MODEL_REGISTRY


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gpt-4o-mini", OpenAIChatClient),
        ("gpt-4o", OpenAIChatClient),
        ("gpt-4.1", OpenAIChatClient),
        ("gpt-5", OpenAIResponsesClient),
        ("gpt-5.5", OpenAIResponsesClient),
        ("gemini-2.5-flash", GeminiClient),
        ("claude-sonnet-4-6", AnthropicClient),
    ],
)
def test_build_routes_to_expected_adapter(model_id: str, expected: type) -> None:
    client = build_llm_client(model_id, response_schema_type="adjustments")
    assert isinstance(client, expected)
    assert client.model == model_id


def test_build_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        build_llm_client("gpt-4-turbo", response_schema_type="adjustments")


def test_gpt5_uses_fixed_temperature_and_gpt55_is_tunable() -> None:
    fixed = build_llm_client(
        "gpt-5", response_schema_type="adjustments", temperature=0.0
    )
    tunable = build_llm_client(
        "gpt-5.5", response_schema_type="adjustments", temperature=0.0
    )
    assert fixed.temperature is None  # omitted for fixed-temp reasoning models
    assert tunable.temperature == 0.0


def test_reasoning_models_get_higher_output_token_floor_and_effort() -> None:
    for model in ("gpt-5", "gpt-5.5"):
        client = build_llm_client(
            model, response_schema_type="vitals", max_output_tokens=300
        )
        # 300 is consumed by reasoning tokens; the floor leaves room for JSON.
        assert client.max_output_tokens >= 4000
        assert client.reasoning_effort is not None


def test_non_reasoning_model_keeps_requested_tokens_and_no_effort() -> None:
    client = build_llm_client(
        "gpt-4o", response_schema_type="vitals", max_output_tokens=300
    )
    assert client.max_output_tokens == 300
    assert client.reasoning_effort is None


def test_caller_can_exceed_reasoning_token_floor() -> None:
    client = build_llm_client(
        "gpt-5.5", response_schema_type="vitals", max_output_tokens=8000
    )
    assert client.max_output_tokens == 8000


def test_missing_key_degrades_to_empty_adjustments_json() -> None:
    client = build_llm_client(
        "gpt-4o", response_schema_type="adjustments", api_key=""
    )
    parsed = json.loads(client.complete("prompt"))
    assert parsed == {"adjustments": {}}
    assert "OPENAI_API_KEY" in client.last_api_error


def test_missing_key_degrades_to_empty_vitals_json() -> None:
    client = build_llm_client(
        "gemini-2.5-flash", response_schema_type="vitals", api_key=""
    )
    parsed = json.loads(client.complete("prompt"))
    assert parsed == {"vitals": {}}
    assert "GEMINI_API_KEY" in client.last_api_error


def test_anthropic_missing_key_degrade_mentions_env_var() -> None:
    client = build_llm_client(
        "claude-sonnet-4-6", response_schema_type="vitals", api_key=""
    )
    client.complete("prompt")
    assert "ANTHROPIC_API_KEY" in client.last_api_error


def test_explicit_empty_key_forces_degrade_regardless_of_env() -> None:
    # api_key="" (not None) is used verbatim, never falling back to os.environ.
    client = OpenAIChatClient(
        model="gpt-4o",
        response_schema_type="adjustments",
        temperature=0.0,
        max_output_tokens=300,
        api_key="",
    )
    json.loads(client.complete("prompt"))  # must not raise / must not call SDK
    assert client.last_api_error is not None
