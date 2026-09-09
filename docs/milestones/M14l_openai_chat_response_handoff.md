# M14l OpenAI Chat Response Handoff

## Scope

Hardened OpenAI-compatible relay response handling in `generate_utterances.py`
after a GPT-5.5 run failed at `resp.choices[0].message.content`.

No scenario JSON, transition engines, runtime state schema, or tests were
modified.

## Problem

The traceback stopped at:

```text
generate_utterances.py, line 243, in generate
return resp.choices[0].message.content or ""
```

The final exception line was not shown, but that location can fail when an
OpenAI-compatible relay returns:

- no `choices`
- a choice without a standard `message`
- empty `message.content`, often with `finish_reason="length"`
- content as a list of text parts instead of a string

GPT reasoning models can also consume a small completion budget before emitting
JSON, so the previous hard-coded `700` token agent budget was fragile.

## Changes

- `generate_utterances.py`
  - Added `_chat_response_text(...)` to extract text from OpenAI-compatible chat
    responses safely.
  - Added clear runtime errors with response summaries for missing choices,
    missing messages, and empty content.
  - Reused `_chat_response_text(...)` for OpenAI emotion JSON calls.
  - Added `AGENT_LLM_MAX_TOKENS`, defaulting to `2000`, for agent LLM calls.

## Verification

```bash
python -m py_compile generate_utterances.py emotion_engine.py
```

Result: passed.

Fake response checks:

- Standard `choices[0].message.content` response returns text.
- Empty `choices` raises `RuntimeError` with response summary.
- Empty content with `finish_reason="length"` raises `RuntimeError` identifying
  empty message content and finish reason.

## Suggested Rerun

Use a larger completion budget for GPT-5.5:

```bash
env AGENT_LLM_MODEL=gpt-5.5 \
AGENT_LLM_MAX_TOKENS=3000 \
AGENT_LLM_MIN_INTERVAL=8 \
AGENT_LLM_RETRY_BASE_SECONDS=60 \
python generate_utterances.py \
  --provider openai \
  --emotion \
  --turns 5 \
  --trait-seed 1 \
  --physiology-mode hybrid \
  --scenario "transitions/Resuscitation/Intra-abdominal Sepsis.json" \
  --output run_sepsis_v3_gpt55_5turns.json
```
