# M5 Agent Base Context

## Goal

Define the agent interface and static/dynamic prompt contract with mock agents only, before real LLM integration.

## Relevant Sections From V4

- Section 9: BaseAgent and role principles.
- Section 13: AI clinician output must be structured.
- Section 23.4: static/dynamic prompt contract.
- Section 24.10: requirements before first LLM agent.

## Allowed Files/Folders

- `ed_multiagent/agents/`
- `ed_multiagent/actions/schema.py`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not call any LLM provider.
- Do not add API clients, API keys, streaming, retries, or network behavior.
- Do not expose hidden state or undiscovered facts in prompts.
- Do not add AI clinician autonomy beyond mocks.

## Required Tests

- Every mock agent implements `build_static_prompt()` and `build_dynamic_prompt(observation)`.
- `act(observation)` returns structured `AgentTurnOutput`.
- Static prompt remains invariant across rounds.
- Dynamic prompt changes with observation and excludes hidden fields.

## Acceptance Criteria

- Prompt boundary exists before real LLM agents.
- Mock agents remain deterministic.
- Future NurseAgent LLM integration can reuse the contract without changing orchestration.
