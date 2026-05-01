# M8 AI Clinician Context

## Goal

Add AI clinician behavior only after deterministic simulation, workflow, memory, observations, validation, logging, and non-clinician agents are stable.

## Relevant V4 Design Sections

- Section 9.2: clinician prompt principles.
- Section 13: AI clinician must output structured actions.
- Section 20.3: validator boundary, unsafe executable actions are allowed and logged.
- Section 23.7: structured code declarations.
- Section 24.1: CODE-mode clinician activation is decision-event driven.
- Section 24.2: unified activation policy separates workflow from LLM activation.

## Current Dependencies

- `ed_multiagent/agents/clinician.py` contains the no-LLM clinician skeleton.
- `ed_multiagent/agents/base.py` provides static/dynamic prompt construction.
- `ed_multiagent/observation/gateway.py` provides `ClinicianObservation` from discovered facts only.
- `ed_multiagent/actions/schema.py` defines `AgentTurnOutput`.
- `ed_multiagent/orchestrator/simulation.py` accepts structured `ClinicianTurn` inputs.
- `ed_multiagent/orchestrator/activation_policy.py` controls CODE/urgent clinician call cadence.
- `ed_multiagent/evaluation/logger.py` records replayable round data.

## Allowed Files/Folders

- `ed_multiagent/agents/clinician.py`
- `ed_multiagent/agents/llm_client.py`
- `ed_multiagent/actions/parser.py`
- `ed_multiagent/actions/schema.py`
- `ed_multiagent/evaluation/scorer.py`
- `tests/ed_multiagent/test_clinician_agent_llm.py`

## Forbidden Scope

- Do not let AI clinician access ground truth, hidden physiology, hidden diagnosis, raw pathology, undiscovered facts, or rule-engine internals.
- Do not accept free text as the only clinical action output.
- Do not let AI clinician directly execute EMSim actions; orders must flow through validation, workflow, nurse tasks, and EMSim action mapping.
- Do not block unsafe but executable orders in the validator; log and score them.
- Do not change EMSim physiology to make AI clinician behavior look better.

## Required Behavior

- AI clinician receives only `ClinicianObservation` and structured discovered memory.
- Output must parse to structured verbal actions, information actions, order bundles, and optional meta actions such as declare/exit code.
- CODE mode calls AI clinician at ACLS decision events or configured cadence, not every 5 seconds.
- Unsafe but executable orders are allowed through workflow, logged, and scored by evaluation.
- Invalid LLM output must be rejected or repaired before it can mutate world state.

## Required Tests

- Mocked AI clinician output parses to structured action bundle.
- Clinician prompt omits hidden state, undiscovered allergy, hidden diagnosis, and rule-engine internals.
- Unsafe but executable order is allowed, logged, and available for scoring.
- CODE decision-event cadence prevents clinician calls every 5 seconds.

## Acceptance Criteria

- AI clinician can run through the existing deterministic simulator without bypassing workflow or memory boundaries.
- Logs remain sufficient to evaluate clinical quality, timing, communication, and safety.
- CI tests use mocked LLM responses and require no live provider access unless a separate explicit integration test is enabled.

## Future Hybrid-Engine Compatibility Constraints

- The current EMSim rule-based engine may later be replaced by a hybrid engine.
- AI clinician must consume only `ObservationGateway` outputs and structured discovered memory, never engine internals.
- Do not import from `rule_engine` or inspect `EngineSession` hidden/private fields.
- Clinician prompts and parser contracts must stay engine-agnostic so a hybrid physiology backend can preserve the same role-facing surface.
