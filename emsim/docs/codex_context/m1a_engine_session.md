# M1a EngineSession Context

## Goal

Wrap EMSim as a live `EngineSession` that persists physiology over time and exposes observable vitals without exposing hidden physiology.

## Relevant Sections From V4

- Section 2: Phase 1, live `EngineSession`.
- Section 2.3: existing EMSim action format for interventions and drugs.
- Section 20.9 and 23.8: M1a no-orders physiology run.
- Section 24.4: future `EngineAdvanceResult` and physiology event contract.

## Allowed Files/Folders

- `ed_multiagent/emsim_adapter/`
- `tests/ed_multiagent/`
- `docs/` for notes if needed
- `rule_engine/session.py` only if explicitly requested for core integration

## Forbidden Scope

- Do not implement workflow, orders, memory, observations, or agents.
- Do not add LLM calls.
- Do not alter physiology formulas, drug PD, intervention effects, pathology drift, or hidden-state semantics.
- Do not invent vitals outside EMSim.

## Required Tests

- Construct session from an initial EMSim schema state.
- Advance multiple rounds and verify time/state persist.
- Apply `apply_NRB`, `give_fluids`, and one drug action through EMSim-compatible action dicts.
- Verify agent-facing observations omit hidden state.

## Acceptance Criteria

- A scenario initial state can run 30 physiology-only rounds.
- Vitals are produced by EMSim and logged over time.
- Existing EMSim transition behavior is unchanged.
