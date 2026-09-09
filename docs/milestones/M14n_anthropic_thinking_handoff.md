# M14n Anthropic Thinking Handoff

## Summary

Added optional Anthropic thinking support to `generate_utterances.py`.

The default behavior is unchanged. Thinking is only enabled when the caller sets
`ANTHROPIC_THINKING_TYPE`.

## Changes

- Added `_anthropic_thinking_kwargs()` to build Anthropic Messages API thinking
  parameters from environment variables.
- Wired the thinking parameters into:
  - `AnthropicRelayClient.generate(...)`
  - `_anthropic_json_call(...)`
- When Anthropic thinking is enabled, emotion JSON calls use
  `AGENT_LLM_MAX_TOKENS` instead of the old fixed `1024`, because thinking
  tokens count toward `max_tokens`.

## Environment Variables

- `ANTHROPIC_THINKING_TYPE`
  - `adaptive`: recommended for Claude Sonnet 4.6.
  - `enabled`: manual thinking with `ANTHROPIC_THINKING_BUDGET_TOKENS`.
  - `disabled`: explicitly disabled.
- `ANTHROPIC_THINKING_EFFORT`
  - Optional. For adaptive thinking, examples include `low`, `medium`, `high`,
    and `max`.
- `ANTHROPIC_THINKING_DISPLAY`
  - Optional Anthropic display mode such as `summarized` or `omitted`.
- `ANTHROPIC_THINKING_BUDGET_TOKENS`
  - Required only when `ANTHROPIC_THINKING_TYPE=enabled`.

## Verification

Ran:

```bash
python -m py_compile generate_utterances.py
ANTHROPIC_THINKING_TYPE=adaptive ANTHROPIC_THINKING_EFFORT=medium python -c 'import generate_utterances as g; print(g._anthropic_thinking_kwargs())'
ANTHROPIC_THINKING_TYPE=enabled ANTHROPIC_THINKING_BUDGET_TOKENS=2000 python -c 'import generate_utterances as g; print(g._anthropic_thinking_kwargs())'
```

No live Anthropic simulation was run in this handoff.
