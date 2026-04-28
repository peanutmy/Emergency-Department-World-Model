# M1b Workflow Engine Context

## Goal

Implement deterministic order and nurse-task workflow so hand-typed orders execute after simulated delays and then call EMSim.

## Relevant Sections From V4

- Section 3: WorldState, OrderManager, NurseTaskQueue, LabScheduler.
- Section 6.4: clinician orders are not immediate world mutations.
- Section 17: MVP round duration and suggested task durations.
- Section 23.2: task duration across SimMode changes.
- Section 23.8: M1b hand-typed orders to EMSim.

## Allowed Files/Folders

- `ed_multiagent/world/`
- `ed_multiagent/emsim_adapter/action_mapper.py`
- `ed_multiagent/actions/schema.py`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not add LLM agents.
- Do not modify EMSim core physiology.
- Do not bypass nurse task delay.
- Do not implement broad clinical safety validation beyond structural needs.

## Required Tests

- Drug order creates task, task completes after delay, EMSim action applies.
- Intervention order maps to existing EMSim action format.
- Lab order schedules result and returns after configured turnaround.
- `remaining_s` decrements by simulated seconds.

## Acceptance Criteria

- Hand-typed JSON orders move through order manager, task queue, action mapper, and `EngineSession`.
- Vitals change only after task execution and EMSim advancement.
- Workflow events are recordable for later logging.
