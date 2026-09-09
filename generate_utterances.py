#!/usr/bin/env python
"""Generate real LLM utterances for a scenario via Claude or an OpenAI relay.

The demo's built-in real agent client targets the OpenAI Responses API, which
many relays do not proxy. This script provides two clients that work with
relays and injects the chosen one into the real-agent demo, then prints the
utterances generated per turn.

Providers
---------
--provider openai     (default) OpenAI-compatible relay, via /v1/chat/completions.
                      Reads OPENAI_API_KEY and OPENAI_BASE_URL.
--provider anthropic  Anthropic (or an Anthropic-style relay), via /v1/messages.
                      Reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL.

Set credentials in your shell (~/.zshrc), e.g. for an OpenAI relay:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="http://<relay-host>:3000/v1"

Model: override with AGENT_LLM_MODEL (else a per-provider default is used).

Args:
    --scenario PATH   scenario JSON (default: first transitions scenario)
    --turns N         number of turns (default: 6)
    --output PATH     also write the full trajectory JSON
    --physiology-mode MODE
                      physiology backend: hybrid (default) or fake.
                      true/real are accepted as aliases for hybrid.
    --list-models     query the relay's /v1/models and exit
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ed_world_model.demo import (
    PHYSIOLOGY_MODE_FAKE,
    PHYSIOLOGY_MODE_HYBRID,
    run_demo_scenario,
)
from emotion_engine import PROMPT_STRATEGIES


DEFAULT_SCENARIO = "transitions/Cardiology/Acute Respiratory Distress.json"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-1-20250805"
INTRA_ABDOMINAL_SEPSIS_CASE_MARKERS = (
    "intra-abdominal sepsis",
    "intra_abdominal_sepsis",
)
PHYSIOLOGY_MODE_ALIASES = {
    PHYSIOLOGY_MODE_FAKE: PHYSIOLOGY_MODE_FAKE,
    "false": PHYSIOLOGY_MODE_FAKE,
    "0": PHYSIOLOGY_MODE_FAKE,
    "no": PHYSIOLOGY_MODE_FAKE,
    PHYSIOLOGY_MODE_HYBRID: PHYSIOLOGY_MODE_HYBRID,
    "true": PHYSIOLOGY_MODE_HYBRID,
    "1": PHYSIOLOGY_MODE_HYBRID,
    "yes": PHYSIOLOGY_MODE_HYBRID,
    "real": PHYSIOLOGY_MODE_HYBRID,
}

# Agent prompts demand strict JSON; keep the model from wrapping it in prose.
_SYSTEM = (
    "You are a backend component in a simulation. Respond with ONLY the final "
    "JSON object the prompt asks for. No markdown fences, no commentary, no "
    "explanation before or after the JSON."
)


def _model_for(provider: str) -> str:
    override = os.environ.get("AGENT_LLM_MODEL")
    if override:
        return override
    return DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_ANTHROPIC_MODEL


# Minimum seconds between any two LLM API calls, to stay under per-minute
# token-rate limits. Override with AGENT_LLM_MIN_INTERVAL (e.g. "3"). 0 disables.
_MIN_CALL_INTERVAL = float(os.environ.get("AGENT_LLM_MIN_INTERVAL", "3"))
_MAX_LLM_ATTEMPTS = int(os.environ.get("AGENT_LLM_MAX_ATTEMPTS", "6"))
_RETRY_BASE_SECONDS = float(os.environ.get("AGENT_LLM_RETRY_BASE_SECONDS", "20"))
_RETRY_MAX_SECONDS = float(os.environ.get("AGENT_LLM_RETRY_MAX_SECONDS", "120"))
# Adaptive thinking counts toward max_tokens; 2000 truncated agent JSON
# mid-object under ANTHROPIC_THINKING_EFFORT=max.
_AGENT_MAX_TOKENS = int(os.environ.get("AGENT_LLM_MAX_TOKENS", "8000"))
_ANTHROPIC_THINKING_TYPE = os.environ.get("ANTHROPIC_THINKING_TYPE", "").strip()
_ANTHROPIC_THINKING_EFFORT = os.environ.get(
    "ANTHROPIC_THINKING_EFFORT", ""
).strip()
_ANTHROPIC_THINKING_DISPLAY = os.environ.get(
    "ANTHROPIC_THINKING_DISPLAY", ""
).strip()
_ANTHROPIC_THINKING_BUDGET_TOKENS = os.environ.get(
    "ANTHROPIC_THINKING_BUDGET_TOKENS", ""
).strip()
_LAST_CALL_AT = [0.0]


def _throttle() -> None:
    """Sleep so consecutive LLM calls are spaced by at least _MIN_CALL_INTERVAL."""
    if _MIN_CALL_INTERVAL <= 0:
        return
    elapsed = time.monotonic() - _LAST_CALL_AT[0]
    wait = _MIN_CALL_INTERVAL - elapsed
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_AT[0] = time.monotonic()


def _is_retryable_llm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if _is_unavailable_model_error(exc):
        return False
    return any(k in msg for k in (
        "429",
        "too_many_requests",
        "rate limit",
        "rate_limit",
        "saturat",
        "overload",
        "upstream",
        "503",
        "502",
        "timeout",
        "connection",
        "reset by peer",
        "read error",
        "apiconnection",
        "remoteprotocol",
        "incomplete",
    ))


def _is_unavailable_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "model_not_found",
        "model not found",
        "no available channel for model",
        "model does not exist",
        "does not exist",
    ))


def _unavailable_model_message(model: str, exc: Exception) -> str:
    return (
        f"Model {model!r} is not available from the configured provider/relay. "
        "Run `python generate_utterances.py --provider openai --list-models` "
        "and set AGENT_LLM_MODEL to one of the listed model ids. "
        f"Original error: {exc}"
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    header_wait = _retry_after_seconds(exc)
    if header_wait is not None:
        return min(header_wait, _RETRY_MAX_SECONDS)
    return min(_RETRY_BASE_SECONDS * (2 ** attempt), _RETRY_MAX_SECONDS)


def _anthropic_messages_create(client, **kwargs):
    last_err = None
    for attempt in range(_MAX_LLM_ATTEMPTS):
        try:
            _throttle()
            return client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if not _is_retryable_llm_error(exc) or attempt >= _MAX_LLM_ATTEMPTS - 1:
                break
            wait = _retry_wait_seconds(exc, attempt)
            print(
                "Anthropic LLM call was rate-limited/transiently unavailable; "
                f"retrying in {wait:.0f}s "
                f"({attempt + 2}/{_MAX_LLM_ATTEMPTS}).",
                file=sys.stderr,
            )
            time.sleep(wait)
            _LAST_CALL_AT[0] = time.monotonic()
    raise last_err


def _anthropic_thinking_kwargs() -> dict[str, object]:
    """Build optional Anthropic thinking params from environment variables.

    Defaults to no thinking params so existing runs keep the same latency/cost.
    For Sonnet 4.6, set ANTHROPIC_THINKING_TYPE=adaptive.
    """

    thinking_type = _ANTHROPIC_THINKING_TYPE.lower()
    if not thinking_type:
        return {}
    if thinking_type not in {"adaptive", "enabled", "disabled"}:
        raise RuntimeError(
            "ANTHROPIC_THINKING_TYPE must be adaptive, enabled, or disabled."
        )

    thinking: dict[str, object] = {"type": thinking_type}
    if _ANTHROPIC_THINKING_DISPLAY:
        thinking["display"] = _ANTHROPIC_THINKING_DISPLAY
    if thinking_type == "enabled":
        if not _ANTHROPIC_THINKING_BUDGET_TOKENS:
            raise RuntimeError(
                "ANTHROPIC_THINKING_BUDGET_TOKENS is required when "
                "ANTHROPIC_THINKING_TYPE=enabled."
            )
        thinking["budget_tokens"] = int(_ANTHROPIC_THINKING_BUDGET_TOKENS)

    kwargs: dict[str, object] = {"thinking": thinking}
    if _ANTHROPIC_THINKING_EFFORT:
        kwargs["output_config"] = {"effort": _ANTHROPIC_THINKING_EFFORT}
    return kwargs


def _normalize_physiology_mode(value: str) -> str:
    key = value.strip().lower()
    try:
        return PHYSIOLOGY_MODE_ALIASES[key]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "physiology mode must be fake, hybrid, true, false, real, yes, no, 1, or 0"
        ) from exc


def _object_get(value, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_summary(value) -> str:
    try:
        if hasattr(value, "model_dump_json"):
            return value.model_dump_json()[:1200]
        if isinstance(value, dict):
            return json.dumps(value, default=str)[:1200]
    except Exception:  # noqa: BLE001
        pass
    return repr(value)[:1200]


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = _object_get(item, "text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def _chat_response_text(resp) -> str:
    """Extract assistant text from OpenAI-compatible chat responses."""

    choices = _object_get(resp, "choices")
    if not choices:
        output_text = _object_get(resp, "output_text")
        if output_text:
            return _content_to_text(output_text)
        raise RuntimeError(
            "OpenAI chat completion returned no choices. "
            f"Response summary: {_response_summary(resp)}"
        )

    choice = choices[0]
    message = _object_get(choice, "message")
    if message is None:
        text = _object_get(choice, "text")
        if text:
            return _content_to_text(text)
        raise RuntimeError(
            "OpenAI chat completion choice had no message. "
            f"Response summary: {_response_summary(resp)}"
        )

    text = _content_to_text(_object_get(message, "content"))
    if text:
        return text

    finish_reason = _object_get(choice, "finish_reason")
    raise RuntimeError(
        "OpenAI chat completion returned empty message content "
        f"(finish_reason={finish_reason!r}). "
        f"Response summary: {_response_summary(resp)}"
    )


def _chat_create(client, model: str, messages: list, max_tokens: int):
    """Call chat.completions, adapting params per model family.

    Reasoning models (gpt-5*, o1/o3/o4) require ``max_completion_tokens`` and
    reject a non-default ``temperature``; classic chat models use ``max_tokens``
    with ``temperature``. Try the expected style first, fall back to the other.
    """
    reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    if reasoning:
        attempts = [
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens, "temperature": 0.0},
        ]
    else:
        attempts = [
            {"max_tokens": max_tokens, "temperature": 0.0},
            {"max_completion_tokens": max_tokens},
        ]
    last_err = None
    max_attempts = 6
    for extra in attempts:
        for attempt in range(max_attempts):  # retry transient relay errors
            try:
                _throttle()
                resp = client.chat.completions.create(
                    model=model, messages=messages, **extra)
                # A gateway hiccup can return 200 with an HTML page (no choices)
                # instead of a chat completion. Treat that as a transient error
                # so it retries rather than crashing the run.
                if not _object_get(resp, "choices") and not _object_get(
                    resp, "output_text"
                ):
                    raise RuntimeError(
                        "relay returned no choices (transient gateway/HTML "
                        f"response): {_response_summary(resp)}"
                    )
                return resp
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if _is_unavailable_model_error(exc):
                    raise RuntimeError(_unavailable_model_message(model, exc)) from exc
                if any(k in msg for k in (
                    "max_tokens", "max_completion_tokens", "temperature",
                    "unsupported_parameter",
                )):
                    break  # param mismatch -> try the alternate param style
                transient = any(k in msg for k in (
                    "429", "too_many_requests", "rate", "saturat", "overload",
                    "upstream", "503", "502", "timeout",
                    "connection", "reset by peer", "read error", "apiconnection",
                    "remoteprotocol", "incomplete",
                    "no choices", "html", "gateway",
                ))
                if transient and attempt < max_attempts - 1:
                    # exponential backoff to let a saturated relay recover
                    time.sleep(min(5 * 2 ** attempt, 60))  # 5,10,20,40,60
                    continue
                break  # other error -> try next param style, else raise
    raise last_err


class OpenAIRelayClient:
    """OpenAI-compatible client via /v1/chat/completions (relay-friendly)."""

    def __init__(self, *, model: str, max_tokens: int | None = None) -> None:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model
        self.max_tokens = max_tokens or _AGENT_MAX_TOKENS
        # OpenAI() reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
        self._client = OpenAI()

    def generate(self, prompt: str) -> str:
        resp = _chat_create(
            self._client, self.model,
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            self.max_tokens,
        )
        return _chat_response_text(resp)

    __call__ = generate

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


class AnthropicRelayClient:
    """Anthropic client via /v1/messages (official API or Anthropic-style relay)."""

    def __init__(self, *, model: str, max_tokens: int | None = None) -> None:
        from anthropic import Anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.max_tokens = max_tokens or _AGENT_MAX_TOKENS
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL

    def generate(self, prompt: str) -> str:
        text = ""
        for attempt in range(2):
            resp = _anthropic_messages_create(
                self._client,
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                **_anthropic_thinking_kwargs(),
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            if text.strip():
                return text
            # Adaptive thinking can consume the whole max_tokens budget and
            # end the response before any text block; one fresh sample
            # usually recovers.
            print(
                f"[llm] empty text response "
                f"(stop_reason={getattr(resp, 'stop_reason', None)}), "
                f"attempt {attempt + 1}/2",
                flush=True,
            )
        return text

    __call__ = generate

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


def _make_client(provider: str, model: str):
    if provider == "openai":
        return OpenAIRelayClient(model=model)
    return AnthropicRelayClient(model=model)


def _extract_json_object(text: str) -> dict:
    """Parse the last balanced top-level JSON object from a model response.

    A naive find('{')..rfind('}') breaks on reasoning/"thinking" models whose
    prose contains braces (e.g. '{fear: low}'), producing a malformed span. This
    brace-matches every balanced {...} block (ignoring braces inside strings)
    and returns the last one that parses as JSON -- the actual answer.
    """
    candidates: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start:i + 1])
    for span in reversed(candidates):
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no parseable JSON object in response: {text[:300]}")


def _openai_json_call(model: str):
    """Return an llm_call(prompt)->dict for the emotion engine via the relay."""
    from openai import OpenAI

    client = OpenAI()

    def call(prompt: str, *, model: str = model) -> dict:
        resp = _chat_create(
            client, model,
            [
                {"role": "system",
                 "content": "Return ONLY a JSON object. No prose, no code fences."},
                {"role": "user", "content": prompt},
            ],
            1024,
        )
        return _extract_json_object(_chat_response_text(resp))

    return call


def _anthropic_json_call(model: str):
    """Return an llm_call(prompt)->dict for the emotion engine via Anthropic."""
    from anthropic import Anthropic

    client = Anthropic()

    def call(prompt: str, *, model: str = model) -> dict:
        resp = _anthropic_messages_create(
            client,
            model=model,
            max_tokens=_AGENT_MAX_TOKENS if _ANTHROPIC_THINKING_TYPE else 1024,
            system="Return ONLY a JSON object. No prose, no code fences.",
            messages=[{"role": "user", "content": prompt}],
            **_anthropic_thinking_kwargs(),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _extract_json_object(text)

    return call


def _make_json_call(provider: str, model: str):
    if provider == "openai":
        return _openai_json_call(model)
    return _anthropic_json_call(model)


def _is_intra_abdominal_sepsis_case(scenario: str, description: str) -> bool:
    haystack = f"{scenario}\n{description}".lower()
    return any(marker in haystack for marker in INTRA_ABDOMINAL_SEPSIS_CASE_MARKERS)


def _add_case_scoped_emotion_debug(
    record: dict,
    *,
    turn_index: int,
    emotion_engine,
    emotion_update_recorded: bool,
) -> None:
    """Write emotion errors only for the intra-abdominal sepsis debug run."""

    last_error = getattr(emotion_engine, "last_error", None)
    record["patient_emotion_last_error"] = last_error
    if not last_error or not emotion_update_recorded:
        return
    record["events"].append(
        {
            "type": "emotion_engine_error",
            "turn_index": turn_index,
            "payload": {
                "agent": "patient_emotion",
                "error": last_error,
            },
        }
    )


def _run_with_emotion(
    provider: str,
    scenario: str,
    turns: int,
    model: str,
    trait_seed,
    output,
    physiology_mode: str,
    prompt_strategy: str,
) -> int:
    """Run the demo with the REAL emotion engine wired for patient + relative."""
    from ed_world_model.scenario_loader import ScenarioLoader
    from ed_world_model.demo import build_demo_agents, build_demo_physiology_adapter
    from ed_world_model.orchestration.runner import IntegrationRunner
    from ed_world_model.adapters.emotion_engine_adapter import EmotionEngineAdapter
    from ed_world_model.state.global_state import patient_is_preverbal

    loader = ScenarioLoader(randomize_traits=True, trait_seed=trait_seed)
    state = loader.load(scenario, max_turns=turns)

    print(f"Provider: {provider} (+ real emotion engine)")
    print(f"Model:    {model}")
    print(f"Physiology: {physiology_mode}")
    print(f"Emotion prompt strategy: {prompt_strategy}")
    print(f"Scenario: {scenario}")
    print(f"Patient traits : {dict(state.agent_profiles.patient.traits)}")
    # Relative emotion/traits are only surfaced if the relative actually speaks.
    print("=" * 72)

    json_call = _make_json_call(provider, model)
    scen = state.truth_state.scenario_description or ""
    debug_patient_emotion_errors = _is_intra_abdominal_sepsis_case(scenario, scen)
    vit = state.patient_state.vitals.model_dump()
    patient_emotion = EmotionEngineAdapter(
        agent_role="patient", scenario=scen, initial_vitals=vit,
        llm_call=json_call, prompt_strategy=prompt_strategy)
    relative_emotion = EmotionEngineAdapter(
        agent_role="relative", scenario=scen, initial_vitals=vit,
        llm_call=json_call, prompt_strategy=prompt_strategy)

    runner = IntegrationRunner(
        state,
        agents=build_demo_agents(
            state, agent_mode="real",
            llm_client_factory=lambda: _make_client(provider, model)),
        physiology_adapter=build_demo_physiology_adapter(
            physiology_mode=physiology_mode,
            llm_client_factory=lambda: _make_client(provider, model),
        ),
        emotion_engine=patient_emotion,
        relative_emotion_engine=relative_emotion,
    )

    out_turns = []
    relative_present = False
    # For a preverbal child the relative is the dialogue partner and the only
    # tracked emotional agent, so being spoken to also counts as presence
    # (mirrors the turn-loop emotion gate).
    patient_preverbal = patient_is_preverbal(runner.state)
    for _ in range(turns):
        tt = runner.run_turn()
        if any(
            m.get("speaker") == "relative"
            or (patient_preverbal and m.get("recipient") == "relative")
            for m in tt.committed_messages
        ):
            relative_present = True
        pe = runner.state.psych_state.patient_emotion
        print(f"-- Turn {tt.turn_index_before}  active={tt.active_agents}")
        for m in tt.committed_messages:
            print(f"   {m.get('speaker')} -> {m.get('recipient') or 'public'}: "
                  f"{m.get('content')}")
        print(f"   [patient ] {pe.summary()}")
        # Surface emotion-engine failures instead of silently freezing emotion.
        emotion_err = getattr(patient_emotion, "last_error", None)
        if emotion_err:
            print(f"   [patient emotion ENGINE ERROR] {emotion_err}")
        emotion_update_recorded = any(
            event.get("type") == "emotion_update"
            and (event.get("payload") or {}).get("agent") == "patient_emotion"
            for event in tt.events
        )
        record = {
            "turn": tt.turn_index_before,
            "active_agents": list(tt.active_agents),
            "messages": list(tt.committed_messages),
            "physiology_action_kind_hint": tt.physiology_action_kind_hint,
            "physiology_action": tt.physiology_action,
            "state_after": tt.state_after,
            "events": list(tt.events),
            "patient_emotion": pe.model_dump(),
        }
        if debug_patient_emotion_errors:
            _add_case_scoped_emotion_debug(
                record,
                turn_index=tt.turn_index_before,
                emotion_engine=patient_emotion,
                emotion_update_recorded=emotion_update_recorded,
            )
        # Only surface relative emotion once the relative is actually in the
        # scene (has spoken). If they never appear, no relative emotion is shown.
        if relative_present:
            re = runner.state.psych_state.relative_emotion
            print(f"   [relative] {re.summary()}")
            record["relative_emotion"] = re.model_dump()
        out_turns.append(record)
        if tt.terminated:
            break

    if output:
        payload = {
            "scenario": scen,
            "physiology_mode": physiology_mode,
            "emotion_prompt_strategy": prompt_strategy,
            "patient_traits": dict(state.agent_profiles.patient.traits),
            "turns": out_turns,
        }
        if relative_present:
            payload["relative_traits"] = dict(state.agent_profiles.relative.traits)
        Path(output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("=" * 72)
        print(f"Written {output}")
    return 0


def _list_models(provider: str) -> int:
    import httpx

    if provider == "openai":
        base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        url = f"{base}/models"
    else:
        base = (os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        url = f"{base}/v1/models"
    print(f"GET {url}")
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        print("HTTP", r.status_code)
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            print(r.text[:3000])
            return 0
        items = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(items, list) and items:
            for m in items:
                print("  -", m.get("id") if isinstance(m, dict) else m)
        else:
            print("(no model list parsed; raw response body below)")
            print(r.text[:3000])
    except Exception as exc:  # noqa: BLE001
        print("request failed:", exc)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--physiology-mode",
        default=PHYSIOLOGY_MODE_HYBRID,
        type=_normalize_physiology_mode,
        metavar="MODE",
        help=(
            "Physiology backend. Use hybrid for real physiology transitions "
            "(default) or fake for deterministic smoke tests. Aliases: "
            "true/real/yes/1=hybrid, false/no/0=fake."
        ),
    )
    parser.add_argument("--list-models", action="store_true",
                        help="Query the relay's /v1/models and exit.")
    parser.add_argument("--emotion", action="store_true",
                        help="Wire the real emotion engine (patient + relative).")
    parser.add_argument(
        "--emotion-prompt-strategy",
        choices=PROMPT_STRATEGIES,
        default="zero_shot",
        help=(
            "Prompt strategy for the real emotion engine: zero_shot, "
            "few_shot, or react."
        ),
    )
    parser.add_argument("--trait-seed", type=int, default=None,
                        help="Seed for reproducible random agent traits.")
    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models(args.provider)

    model = _model_for(args.provider)

    if args.emotion:
        return _run_with_emotion(args.provider, args.scenario, args.turns, model,
                                 args.trait_seed, args.output,
                                 args.physiology_mode,
                                 args.emotion_prompt_strategy)
    print(f"Provider: {args.provider}")
    print(f"Model:    {model}")
    print(f"Physiology: {args.physiology_mode}")
    print(f"Scenario: {args.scenario}")
    print("=" * 72)

    result = run_demo_scenario(
        args.scenario,
        turns=args.turns,
        agent_mode="real",
        physiology_mode=args.physiology_mode,
        llm_client_factory=lambda: _make_client(args.provider, model),
    )

    for turn in result.turns:
        print(f"-- Turn {turn.turn_index}  active={turn.active_agents}")
        if not turn.messages:
            print("   (no utterances)")
        for m in turn.messages:
            spk = m.get("speaker")
            rcp = m.get("recipient") or "public"
            print(f"   {spk} -> {rcp}: {m.get('content')}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=" * 72)
        print(f"Full trajectory written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
