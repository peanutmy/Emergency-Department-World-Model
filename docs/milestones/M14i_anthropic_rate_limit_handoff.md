# M14i Anthropic Rate Limit Handoff

## Scope

Added Anthropic retry/backoff handling in `generate_utterances.py` for 429 rate
limit and transient availability errors.

No scenario JSON, transition engine, runtime state schema, or tests were
modified.

## Changes

- `generate_utterances.py`
  - Added configurable retry settings:
    - `AGENT_LLM_MAX_ATTEMPTS` default `6`
    - `AGENT_LLM_RETRY_BASE_SECONDS` default `20`
    - `AGENT_LLM_RETRY_MAX_SECONDS` default `120`
  - Added `_anthropic_messages_create(...)`, which wraps
    `client.messages.create(...)` with:
    - existing global LLM throttling
    - retryable error detection for `429`, rate limit, overload, upstream,
      `502`, `503`, and timeout errors
    - `retry-after` header support when present
    - exponential backoff otherwise
    - stderr retry progress messages
  - Wired the wrapper into both Anthropic agent calls and Anthropic emotion JSON
    calls.

## Behavior Notes

- This addresses Anthropic `RateLimitError: 429` during long real-agent runs.
- It does not reduce prompt size. If the org token-per-minute limit remains
  saturated by other traffic, the run may still fail after the configured number
  of attempts.
- For stricter throttling, run with a larger interval, for example:

```bash
AGENT_LLM_MIN_INTERVAL=8 AGENT_LLM_RETRY_BASE_SECONDS=60 python generate_utterances.py ...
```

## Verification

```bash
python -m py_compile generate_utterances.py
```

Result: passed.

```bash
python -c "import generate_utterances as g; e=Exception('429 rate limit'); print(g._is_retryable_llm_error(e)); print(g._retry_wait_seconds(e, 0)); print(g._retry_wait_seconds(e, 2))"
```

Result:

```text
True
20.0
80.0
```

Fake Anthropic client retry check:

```text
first two calls raise "429 rate limit"; third call returns ok
```

Result: wrapper retried and returned `ok` after 3 calls.
