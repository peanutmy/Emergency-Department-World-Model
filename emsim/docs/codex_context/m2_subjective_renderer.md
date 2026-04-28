# M2 Subjective Renderer Context

## Goal

Render current patient subjective experience from EMSim physiology for patient behavior, response capacity, and observation building.

## Relevant Sections From V4

- Section 20.2: Subjective State Renderer.
- Section 20.2: `SubjectiveState` schema and `subjective_from_hidden`.
- Section 23.6: `SubjectiveState` versus `SymptomTimeline`.
- Section 23.3.1: patient activation in CODE depends on speech capacity.

## Allowed Files/Folders

- `ed_multiagent/emsim_adapter/subjective.py`
- `ed_multiagent/memory/private_memory.py`
- `ed_multiagent/observation/`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not expose raw hidden state to agents.
- Do not let the patient decide speech capacity when physiology makes speech impossible.
- Do not modify EMSim core unless explicitly requested.
- Do not add LLM calls.

## Required Tests

- Low consciousness or severe hypoxia reduces speech capacity.
- Tachyarrhythmia physiology yields palpitations.
- Low blood pressure/perfusion yields dizziness.
- Patient-facing output contains subjective fields, not hidden variables.

## Acceptance Criteria

- `SubjectiveState` is deterministic and derived from EMSim state.
- Longitudinal history remains in private memory or symptom timeline.
- Response logic can force `UNABLE` when speech capacity is unable.
