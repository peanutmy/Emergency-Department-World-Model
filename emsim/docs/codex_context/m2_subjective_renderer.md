# M2 Subjective Renderer Context

## Goal

Render current patient subjective experience from EMSim physiology for patient behavior, response capacity, and observation building.

M2 should expose an engine-agnostic renderer first. The current EMSim rule-based
`HiddenState` may be adapted by a narrow compatibility function, but upper
multi-agent layers should consume `SubjectiveState` and should not depend on
rule-engine internals.

## Relevant Sections From V4

- Section 20.2: Subjective State Renderer.
- Section 20.2: `SubjectiveState` schema and `subjective_from_hidden`.
- Section 23.6: `SubjectiveState` versus `SymptomTimeline`.
- Section 23.3.1: patient activation in CODE depends on speech capacity.

## Allowed Files/Folders

- `ed_multiagent/observation/subjective.py`
- `ed_multiagent/observation/__init__.py`
- `ed_multiagent/memory/private_memory.py`
- `ed_multiagent/observation/`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not expose raw hidden state to agents.
- Do not let the patient decide speech capacity when physiology makes speech impossible.
- Do not modify EMSim core unless explicitly requested.
- Do not add LLM calls.
- Do not implement PatientAgent, RelativeAgent, ObservationGateway, SimMode,
  memory updates, or orchestrator behavior in M2.

## Engine Compatibility Boundary

- Primary path: `subjective_from_observation(vitals, rhythm=None, engine_public_state=None)`.
- Portable inputs: vitals, optional rhythm string, optional public engine state
  such as `consciousness`.
- Optional compatibility adapter: `subjective_from_hidden(h, vitals)` may
  duck-type current EMSim fields and call the primary renderer.
- Do not import `rule_engine` symbols from the subjective renderer.
- Do not inspect `EngineSession` private fields.

## Required Tests

- Low consciousness or severe hypoxia reduces speech capacity.
- Tachyarrhythmia physiology yields palpitations.
- Low blood pressure/perfusion yields dizziness.
- Patient-facing output contains subjective fields, not hidden variables.

## Acceptance Criteria

- `SubjectiveState` is deterministic and derived from EMSim state.
- Longitudinal history remains in private memory or symptom timeline.
- Response logic can force `UNABLE` when speech capacity is unable.
