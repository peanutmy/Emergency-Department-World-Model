from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transition_engines.hybrid_engine import HybridEngine
import transition_engines.openai_llm_client as openai_client_module
from transition_engines.openai_llm_client import (
    EMPTY_ADJUSTMENTS_RESPONSE,
    EMPTY_VITALS_RESPONSE,
    MISSING_API_KEY_MESSAGE,
    OpenAILLMClient,
)
from transition_engines.pure_llm_engine import PureLLMEngine
from transition_engines.rule_based import RuleBasedEngine


def _engine_input() -> dict:
    return {
        "case_id": "case-1",
        "category": "synthetic",
        "scenario_description": "Synthetic case.",
        "case_context": {},
        "pair_id": "p1",
        "before": {
            "vitals": {
                "HR": 100,
                "BP_sys": 120,
                "BP_dia": 70,
                "RR": 20,
                "O2Sat": 94,
                "T": 37.0,
            }
        },
        "action": {
            "raw_text": "no action",
            "kind_hint": "no_action",
            "params": {"elapsed_min": 0},
        },
    }


class MockResponses:
    def __init__(self, response: object | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class MockOpenAIClient:
    def __init__(self, responses: MockResponses) -> None:
        self.responses = responses


def test_missing_openai_api_key_raises_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=MISSING_API_KEY_MESSAGE):
        OpenAILLMClient(client=MockOpenAIClient(MockResponses()))


def test_real_client_complete_returns_text_from_mocked_sdk_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    responses = MockResponses(
        response=SimpleNamespace(output_text='{"adjustments": {"HR": 1}}')
    )
    sdk_client = MockOpenAIClient(responses)
    client = OpenAILLMClient(
        model="gpt-test",
        temperature=0.2,
        max_output_tokens=123,
        client=sdk_client,
    )

    result = client.complete("Return JSON only.")

    assert result == '{"adjustments": {"HR": 1}}'
    assert client.last_api_error is None
    assert responses.calls == [
        {
            "model": "gpt-test",
            "input": "Return JSON only.",
            "temperature": 0.2,
            "max_output_tokens": 123,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "hybrid_vital_adjustments",
                    "strict": False,
                    "schema": ANY,
                }
            },
        }
    ]
    assert "test-api-key" not in json.dumps(responses.calls)


def test_real_client_explicit_adjustments_schema_type_uses_adjustments_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    responses = MockResponses(
        response=SimpleNamespace(output_text='{"adjustments": {"HR": 1}}')
    )
    client = OpenAILLMClient(
        response_schema_type="adjustments",
        client=MockOpenAIClient(responses),
    )

    result = client.complete("prompt")

    schema_format = responses.calls[0]["text"]["format"]
    assert result == '{"adjustments": {"HR": 1}}'
    assert schema_format["name"] == "hybrid_vital_adjustments"
    assert schema_format["schema"]["required"] == ["adjustments"]
    assert "adjustments" in schema_format["schema"]["properties"]
    assert "reasoning" in schema_format["schema"]["properties"]
    assert "vitals" not in schema_format["schema"]["properties"]


def test_real_client_vitals_schema_type_uses_vitals_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    responses = MockResponses(
        response=SimpleNamespace(output_text='{"vitals": {"O2Sat": 88}}')
    )
    client = OpenAILLMClient(
        response_schema_type="vitals",
        client=MockOpenAIClient(responses),
    )

    result = client.complete("prompt")

    schema_format = responses.calls[0]["text"]["format"]
    vitals_schema = schema_format["schema"]["properties"]["vitals"]
    assert result == '{"vitals": {"O2Sat": 88}}'
    assert schema_format["name"] == "pure_llm_vital_predictions"
    assert schema_format["schema"]["required"] == ["vitals"]
    assert "vitals" in schema_format["schema"]["properties"]
    assert "reasoning" in schema_format["schema"]["properties"]
    assert "adjustments" not in schema_format["schema"]["properties"]
    assert vitals_schema["required"] == ["HR", "BP_sys", "BP_dia", "RR", "O2Sat", "T"]


def test_invalid_response_schema_type_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Unsupported response_schema_type"):
        OpenAILLMClient(
            response_schema_type="unknown",
            client=MockOpenAIClient(MockResponses()),
        )


def test_real_client_initializes_official_sdk_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    observed: dict[str, object] = {}

    class SpyOpenAI:
        def __init__(self, *, api_key: str) -> None:
            observed["api_key"] = api_key
            self.responses = MockResponses(
                response=SimpleNamespace(output_text='{"adjustments": {}}')
            )

    monkeypatch.setattr(openai_client_module, "OpenAI", SpyOpenAI)

    OpenAILLMClient()

    assert observed == {"api_key": "test-api-key"}


def test_api_error_returns_empty_adjustments_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "test-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    responses = MockResponses(exc=RuntimeError(f"request failed for {api_key}"))
    client = OpenAILLMClient(client=MockOpenAIClient(responses))

    result = client.complete("prompt")

    assert result == EMPTY_ADJUSTMENTS_RESPONSE
    assert json.loads(result) == {"adjustments": {}}
    assert client.last_api_error is not None
    assert api_key not in client.last_api_error
    assert "[REDACTED_OPENAI_API_KEY]" in client.last_api_error


def test_pure_llm_engine_uses_vitals_schema_openai_client_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    responses = MockResponses(
        response=SimpleNamespace(output_text='{"vitals": {"O2Sat": 88}}')
    )
    llm_client = OpenAILLMClient(
        model="gpt-test",
        response_schema_type="vitals",
        client=MockOpenAIClient(responses),
    )
    engine = PureLLMEngine(llm_client)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    assert output["prediction"]["vitals"]["O2Sat"] == 88
    assert output["prediction"]["vitals"]["HR"] == 100
    assert output["metadata"]["llm_reasoning"] == {}
    assert output["metadata"]["llm_parse_error"] is None
    assert output["metadata"]["llm_api_error"] is None
    assert responses.calls[0]["text"]["format"]["schema"]["required"] == ["vitals"]


def test_pure_llm_metadata_includes_redacted_openai_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "test-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    responses = MockResponses(exc=RuntimeError(f"request failed for {api_key}"))
    llm_client = OpenAILLMClient(
        model="gpt-test",
        response_schema_type="vitals",
        client=MockOpenAIClient(responses),
    )
    engine = PureLLMEngine(llm_client)

    output = engine.predict(_engine_input(), target_vital_names=["O2Sat"])

    metadata = output["metadata"]
    serialized_metadata = json.dumps(metadata, sort_keys=True)
    assert api_key not in serialized_metadata
    assert metadata["llm_raw_response"] == EMPTY_VITALS_RESPONSE
    assert metadata["llm_parse_error"] is None
    assert metadata["llm_api_error"] is not None
    assert "[REDACTED_OPENAI_API_KEY]" in metadata["llm_api_error"]


def test_no_api_key_appears_in_hybrid_metadata_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "test-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    responses = MockResponses(exc=RuntimeError(f"request failed for {api_key}"))
    llm_client = OpenAILLMClient(
        model="gpt-test",
        client=MockOpenAIClient(responses),
    )
    engine = HybridEngine(RuleBasedEngine(), llm_client)

    output = engine.predict(_engine_input(), target_vital_names=["HR"])

    metadata = output["metadata"]
    serialized_metadata = json.dumps(metadata, sort_keys=True)
    assert api_key not in serialized_metadata
    assert metadata["llm_raw_response"] == EMPTY_ADJUSTMENTS_RESPONSE
    assert metadata["llm_api_error"] is not None
    assert "[REDACTED_OPENAI_API_KEY]" in metadata["llm_api_error"]
    assert metadata["model"] == "gpt-test"
