# M4 No-LLM Orchestrator Context

## Goal

Connect engine session, workflow, memory, response opportunities, SimMode, observations, mock agents, and logging into a deterministic no-LLM round loop.

## Relevant Sections From V4

- Section 10: main orchestrator loop.
- Section 15: no-LLM world milestone and lab/workflow delay.
- Section 18: avoid four autonomous agents acting freely.
- Section 23.1 and 23.3: dynamic time step and activation policy.
- Section 24.4: physiology events to orchestrator.

## Allowed Files/Folders

- `ed_multiagent/orchestrator/`
- `ed_multiagent/main.py`
- `ed_multiagent/agents/mock.py`
- `ed_multiagent/evaluation/logger.py`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not add real LLM calls.
- Do not bypass order validation, nurse task queue, or EMSim action execution.
- Do not let agents invent vitals or hidden physiology.
- Do not implement autonomous all-agent free-for-all behavior.

## Required Tests

- Hypoxia case: clinician orders NRB, nurse task executes, EMSim changes vitals, log records it.
- Lab case: order, draw, delay, result, discovered memory update.
- Question case: response opportunity, hardcoded answer, discovered memory update.
- CODE case: dynamic time step and activation policy are respected.

## Acceptance Criteria

- A deterministic case runs end to end without LLMs.
- The loop follows clinician turn plus environment reaction.
- All key events are loggable and replayable.
