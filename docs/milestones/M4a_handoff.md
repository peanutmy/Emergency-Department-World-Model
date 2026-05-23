# M4a Handoff

## Files Created or Modified

- `ed_world_model/orchestration/orchestrator.py`
  - Added `OrchestratorDecision`, a lightweight decision object containing:
    - `active_agents`
    - `activation_reasons`
    - `required_response_agents`
  - Added `Orchestrator.select_active_agents(...)`.
  - Added module-level `select_active_agents(...)` convenience function.
  - Implemented deterministic turn-start active-agent selection:
    - clinician is always active with `default_clinician_active`
    - agents in `runtime_state.required_response_agents` are active with
      `required_response`
    - nurse is active for non-empty `runtime_state.newly_available_results`
      with `newly_available_results`
    - relative can be active through explicit selection with
      `explicitly_selected`
  - Deduplicates active agents and activation reasons.
  - Reads `GlobalState` only; it does not mutate state.
- `tests/test_orchestrator.py`
  - Added 15 focused M4a tests covering default clinician activation, required
    responses, nurse result activation, patient/nurse/relative activation rules,
    duplicate prevention, activation reasons, no nurse bedside verbal slot
    selection, no state mutation, and no coupling to transition engines,
    StateManager, or observation building.
- `docs/milestones/M4a_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `ed_world_model/state/state_manager.py`
- `ed_world_model/actions/registry.py`
- `ed_world_model/actions/validator.py`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

No ObservationBuilder, agents, adapters, engine calls, turn loop, pending
question mutation, nurse bedside verbal slot, or GlobalState mutation behavior
was implemented.

## Tests Run and Results

```bash
python -m pytest tests/test_orchestrator.py
```

Result: passed.

```text
15 passed in 0.10s
```

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_action_registry.py tests/test_action_validator.py tests/test_orchestrator.py
```

Result: passed.

```text
118 passed in 0.12s
```

Broad pytest was not run because M4a only adds Orchestrator behavior and the
focused plus adjacent milestone suites passed.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- Explicit selection is scoped to `relative`, matching the Orchestrator policy
  that relative is active only when required to respond or explicitly selected.
  Nurse and patient activation remain limited to their documented turn-start
  conditions.

## Known Limitations

- Orchestrator only selects turn-start active agents and reasons.
- It does not create or resolve `pending_questions`; it only reads
  `runtime_state.required_response_agents`.
- It does not inspect current-turn clinician proposals.
- It does not decide agent content, validate actions, build observations, call
  engines, call adapters, call StateManager, or mutate GlobalState.
- It does not create the nurse bedside verbal slot. That remains a later
  turn-loop reactive slot after a valid clinician medical treatment order.
- It supports the canonical v1.3.1 agent names: `clinician`, `nurse`,
  `patient`, and `relative`.

## Exact Next Step

M4b ObservationBuilder.
