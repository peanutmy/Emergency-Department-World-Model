# M6 Nurse LLM Context

## Goal

Integrate the nurse as the first real LLM-backed agent while keeping nurse task execution deterministic through workflow systems.

## Relevant V4 Design Sections

- Section 9.3: Nurse Agent responsibilities and prompt principles.
- Section 20.7: static prompt caching and cost/latency planning.
- Section 23.3.3: nurse task execution versus nurse LLM verbalization in CODE mode.
- Section 23.4 and 24.10: static/dynamic prompt contract before first LLM agent.
- Section 24.2: workflow execution is separate from optional LLM activation.

## Current Dependencies

- `ed_multiagent/agents/base.py` defines `BaseAgent.build_static_prompt()`, `build_dynamic_prompt(observation)`, and structured `act()`.
- `ed_multiagent/agents/nurse.py` contains the no-LLM `NurseAgent` skeleton.
- `ed_multiagent/actions/schema.py` defines `AgentTurnOutput`.
- `ed_multiagent/observation/gateway.py` and `role_views.py` define `NurseObservation`.
- `ed_multiagent/orchestrator/activation_policy.py` controls when nurse verbalization should be called.
- `ed_multiagent/world/` owns orders, tasks, delays, and EMSim action execution.

## Allowed Files/Folders

- `ed_multiagent/agents/nurse.py`
- `ed_multiagent/agents/llm_client.py`
- `ed_multiagent/config.py`
- `ed_multiagent/actions/schema.py` only for structured output parsing needs
- `tests/ed_multiagent/test_nurse_agent_llm.py`

## Forbidden Scope

- Do not let NurseAgent execute tasks outside `WorkflowEngine` or `NurseTaskQueue`.
- Do not let NurseAgent invent vitals, test results, task status, diagnoses, or hidden physiology.
- Do not expose `rule_engine` internals, hidden state, drug PD internals, or raw pathology to the nurse prompt.
- Do not add patient, relative, or clinician LLM behavior.
- Do not modify EMSim physiology or workflow timing to suit LLM output.

## Required Behavior

- Nurse LLM may communicate only: report observable vitals, warn about observed deterioration, ask clarifying questions, and explain assigned task status.
- Nurse task execution remains deterministic and must continue even when nurse LLM activation is skipped.
- LLM output must be parsed, validated, and returned as `AgentTurnOutput`.
- Invalid or unparseable LLM output must be rejected or repaired without mutating world, workflow, memory, or EMSim state.
- In CODE mode, call nurse LLM only when `AgentActivationPolicy` allows verbal/cognitive activation.

## Required Tests

- Mocked LLM response parses to a structured nurse `AgentTurnOutput`.
- Invalid LLM output is rejected or repaired without state mutation.
- Nurse prompt contains `NurseObservation` data and omits hidden diagnosis, hidden physiology, and rule-engine internals.
- CODE-mode policy can skip nurse LLM while `NurseTaskQueue` execution continues.

## Acceptance Criteria

- Nurse LLM affects communication only.
- Workflow execution remains deterministic, replayable, and testable without network access.
- Tests use mocked LLM responses; real provider calls are gated by explicit configuration and are not required for CI.

## Future Hybrid-Engine Compatibility Constraints

- The current EMSim rule-based engine may later be replaced by a hybrid engine.
- NurseAgent must consume only `ObservationGateway` outputs and structured memory/workflow views.
- Do not import from `rule_engine` or inspect `EngineSession` private fields in nurse agent code.
- Prompts must describe engine-agnostic observations, not current rule-engine field names.
