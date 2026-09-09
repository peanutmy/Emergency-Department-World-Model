# M14m Model Unavailable Handoff

## Scope

Improved `generate_utterances.py` handling for relay errors where the requested
model is unavailable.

No scenario JSON, transition engines, runtime state schema, or tests were
modified.

## Problem

The OpenAI-compatible relay returned:

```text
503 model_not_found: No available channel for model gpt-5.5 under group openai官
```

The script previously treated any `503` as a transient retryable relay error.
For `model_not_found` / `no available channel`, retrying is not useful because
the selected model is not available from the configured relay group.

The shell also reported:

```text
zsh: command not found: --physiology-mode
```

That is a command-line formatting issue, usually caused by a missing trailing
backslash before the `--physiology-mode` line.

## Changes

- `generate_utterances.py`
  - Added `_is_unavailable_model_error(...)`.
  - Excluded unavailable-model errors from retryable LLM errors.
  - In OpenAI chat completion calls, unavailable-model errors now raise a clear
    `RuntimeError` immediately with instructions to run `--list-models` and set
    `AGENT_LLM_MODEL` to an available model id.

## Verification

```bash
python -m py_compile generate_utterances.py emotion_engine.py
```

Result: passed.

Fake error classification:

```text
503 model_not_found no available channel for model gpt-5.5
```

Result:

```text
_is_unavailable_model_error -> True
_is_retryable_llm_error -> False
```
