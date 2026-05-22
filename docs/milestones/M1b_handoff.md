# M1b Handoff

## Files Created or Modified

- `ed_world_model/constants.py`
  - Added `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2`.
  - Did not add a `TEST_TURNAROUND_TURNS` dictionary.
- `ed_world_model/state/global_state.py`
  - Re-exported `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS` by importing it from
    `ed_world_model.constants`, preserving the M1a import surface while keeping
    runtime constants in the shared constants module.
- `ed_world_model/state/state_manager.py`
  - Added `StateManager`, which owns a single `GlobalState` instance and exposes
    it through the `state` property.
  - Added explicit mutate methods:
    - `add_message`
    - `record_event`
    - `update_known_facts`
    - `create_pending_diagnostic_result`
    - `release_ready_diagnostic_results`
    - `apply_physiology_update`
    - `update_patient_emotion`
    - `advance_turn`
  - Diagnostic ordering reads turnaround from the matching `test_bank` item and
    falls back to `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS` when missing.
  - Diagnostic release is the only StateManager method that adds diagnostic
    results to `known_facts.available_results`.
  - No `TurnDelta`, async logic, task queue, plugin framework, retries, or extra
    manager was added.
- `tests/test_state_manager.py`
  - Added 18 focused M1b tests covering message/event separation, known fact
    mutation boundaries, diagnostic pending/release behavior, turn advancement,
    physiology and emotion update boundaries, and absence of `raw_text`.
- `docs/milestones/M1b_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

No ActionRegistry, ActionValidator, Orchestrator, ObservationBuilder, adapters,
agents, or turn loop code was implemented.

## Tests Run and Results

```bash
python -m pytest tests/test_state_manager.py
```

Result: passed.

```text
18 passed in 0.10s
```

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py
```

Result: passed.

```text
30 passed in 0.07s
```

Broad pytest was not run because M1b only adds StateManager behavior and focused
state tests, and the requested focused plus adjacent GlobalState tests passed.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- The design doc shows example StateManager method names, including
  `release_diagnostic_result(test_name, result)`. M1b implements
  `release_ready_diagnostic_results(current_turn=None)` to match the milestone
  request and the turn-loop behavior of releasing all pending results with
  `ready_at_turn <= current_turn`.
- `apply_physiology_update` and `update_patient_emotion` support explicit
  `record_event=True`, but do not record events by default. This follows the M1b
  boundary that they update only `patient_state` or `psych_state.patient_emotion`
  unless event recording is explicitly requested.

## Known Limitations

- `StateManager` is the intended only writer by architecture, but Python callers
  with direct access to `manager.state` can still mutate the underlying
  `GlobalState`. No immutability wrapper was added in M1b.
- `update_known_facts` only appends disclosed history, allergy, medication, and
  symptom facts. It intentionally does not accept diagnostic results; released
  diagnostics enter known facts only through `release_ready_diagnostic_results`.
- Duplicate diagnostic orders are not deduplicated.
- Duplicate `truth_state.test_bank` names are not resolved specially; the first
  matching test name is used.
- `Event.payload` remains generic metadata. StateManager-created events do not
  create `raw_text`, but arbitrary caller-provided event payload metadata is not
  recursively inspected.
- Action registry, action validation, orchestration, observation building,
  adapters, agents, trajectory logging, and the turn loop remain unimplemented.

## M1b_patch

- Runtime event semantics were corrected:
  - `runtime_state.current_turn_events` holds simulator/system events recorded
    during the current turn.
  - `runtime_state.last_turn_events` holds events from the just-completed
    previous turn, for next-turn observations.
  - `StateManager.record_event(...)` now appends to `current_turn_events`.
  - `StateManager.advance_turn()` increments `turn_index`, moves
    `current_turn_events` into `last_turn_events`, clears
    `current_turn_events`, and clears `newly_available_results`.
- Added tests for current-turn event recording, event promotion on
  `advance_turn`, visibility of `last_turn_events`, clearing
  `newly_available_results`, explicit event recording for physiology/emotion
  updates, `record_event=False` behavior, update guard errors, partial emotion
  updates, zero-turnaround diagnostics, multiple ready diagnostic releases, and
  `diagnostic_release.payload["released_at_turn"]`.
- Added a `release_ready_diagnostic_results` docstring clarifying that returned
  results are also stored in `runtime_state.newly_available_results`, so callers
  should avoid double-counting them.
- Deferred pending-question mutate methods to M4a.
- Patch test run:

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py
```

Result: passed.

```text
41 passed in 0.07s
```

- Next step remains M2a ActionRegistry.

## Exact Next Step

M2a ActionRegistry.
