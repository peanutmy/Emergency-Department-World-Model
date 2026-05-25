from __future__ import annotations

from types import SimpleNamespace

import pytest

from ed_world_model.agents.llm_client import (
    FakeLLMClient,
    MISSING_OPENAI_API_KEY_MESSAGE,
    OpenAICompatibleLLMClient,
    RealLLMClient,
)


class MockResponses:
    def __init__(self, *, response: object | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class MockOpenAIClient:
    def __init__(self, responses: MockResponses) -> None:
        self.responses = responses


def test_fake_llm_client_returns_scripted_output_and_default() -> None:
    client = FakeLLMClient(
        [{"verbal_action": None, "action": None}],
        default_response={"verbal_action": None},
    )

    assert client.generate("prompt one") == '{"verbal_action": null, "action": null}'
    assert client("prompt two") == '{"verbal_action": null}'
    assert client.complete("prompt three") == '{"verbal_action": null}'
    assert client.prompts == ["prompt one", "prompt two", "prompt three"]
    assert client.prompt_lengths == [10, 10, 12]
    assert client.last_api_error is None


def test_real_llm_client_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=MISSING_OPENAI_API_KEY_MESSAGE):
        RealLLMClient(client=MockOpenAIClient(MockResponses()))


def test_real_llm_client_generate_uses_mocked_responses_api(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    responses = MockResponses(
        response=SimpleNamespace(output_text='{"verbal_action": null, "action": null}')
    )
    client = OpenAICompatibleLLMClient(
        model="gpt-test",
        temperature=0.2,
        max_output_tokens=123,
        client=MockOpenAIClient(responses),
    )

    result = client.generate("Return JSON only.")

    assert result == '{"verbal_action": null, "action": null}'
    assert client.complete("Return JSON only.") == result
    assert responses.calls[0] == {
        "model": "gpt-test",
        "input": "Return JSON only.",
        "temperature": 0.2,
        "max_output_tokens": 123,
    }
    assert "test-api-key" not in str(responses.calls)


def test_real_llm_client_redacts_api_key_from_exception(monkeypatch) -> None:
    api_key = "secret-test-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    responses = MockResponses(exc=RuntimeError(f"request failed for {api_key}"))
    client = OpenAICompatibleLLMClient(client=MockOpenAIClient(responses))

    with pytest.raises(RuntimeError) as exc_info:
        client.generate("prompt")

    message = str(exc_info.value)
    assert api_key not in message
    assert api_key not in (client.last_api_error or "")
    assert "[REDACTED_OPENAI_API_KEY]" in message
    assert "[REDACTED_OPENAI_API_KEY]" in (client.last_api_error or "")
