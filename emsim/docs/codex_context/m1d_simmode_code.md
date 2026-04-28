# M1d SimMode And CODE Context

## Goal

Implement stateful SimMode, dynamic time steps, CODE-mode activation policy, slash commands, and ROSC event contract scaffolding without LLM calls.

## Relevant Sections From V4

- Section 23.1: stateful `SimModeManager`.
- Section 23.2: task duration in simulated seconds.
- Section 23.3: CODE-mode agent activation policy.
- Section 23.7: structured code declaration.
- Section 24.1: clinician activation policy.
- Section 24.2: centralized `AgentActivationPolicy`.
- Section 24.4: ROSC signal chain and `EngineAdvanceResult`.
- Sections 24.5, 24.6, 24.9: slash commands, hysteresis fixture, M1d checklist.

## Allowed Files/Folders

- `ed_multiagent/orchestrator/`
- `ed_multiagent/actions/`
- `ed_multiagent/world/`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not add LLM calls.
- Do not modify EMSim physiology to force mode transitions.
- Do not implement full ACLS reasoning beyond timers/events required by tests.
- Do not block unsafe but executable clinical actions.

## Required Tests

- Hysteresis fixture from v4 passes.
- Arrest rhythm and `/code` enter CODE.
- `/exit_code` without ROSC does not exit CODE.
- Mock ROSC plus `/exit_code` downgrades CODE to urgent.
- Dynamic `dt_s` is stable: 30s stable, 10s urgent, 5s code.
- Patient, relative, nurse, and clinician activation policies match v4.

## Acceptance Criteria

- SimMode is first-class state, not a stateless helper.
- Workflow and physiology can advance every 5s in CODE while agent cognition is event-triggered.
- Task timing remains measured in simulated seconds.
