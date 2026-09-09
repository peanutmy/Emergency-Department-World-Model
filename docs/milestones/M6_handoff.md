# M6 Handoff

## Files Created or Modified

- `ed_world_model/agents/schemas.py`
  - Added lightweight final-output schemas:
    - `VerbalAction`
    - `AgentProposal`
    - `Agent` protocol
  - `VerbalAction` contains `speaker`, optional `recipient`, `content`, and
    `requires_response`.
  - No internal reasoning or chain-of-thought fields are modeled.
- `ed_world_model/agents/stubs.py`
  - Added deterministic stub/scripted agents:
    - `SilentAgent`
    - `ScriptedClinicianAgent`
    - `ScriptedPatientAgent`
    - `ScriptedNurseAgent`
    - `ScriptedRelativeAgent`
  - Non-clinician scripted agents return verbal output only.
  - Stubs record received observations for focused turn-loop tests.
- `ed_world_model/state/termination.py`
  - Added simple v1.3.1 termination helpers:
    - `is_terminated(global_state)`
    - `termination_reason(global_state)`
  - Termination is `turn_index >= max_turns` or
    `patient_state.status_flags.is_alive is False`.
- `ed_world_model/orchestration/turn_loop.py`
  - Added injected `TurnLoop` and compact `TurnResult`.
  - Coordinates:
    - diagnostic release before selection
    - termination check
    - Orchestrator active-agent selection
    - ObservationBuilder active-only observations
    - stub agent proposal generation
    - clinician action validation
    - verbal-message commits through `StateManager.add_message`
    - diagnostic pending-result creation
    - nurse shadow execution event
    - optional reactive nurse bedside verbal slot after valid treatment
    - `NoopEmotionEngine`-compatible emotion update when messages are committed
    - injected physiology adapter call
    - system-generated `no_action` when no valid treatment exists
    - physiology state application through `StateManager.apply_physiology_update`
    - turn advancement through `StateManager.advance_turn`
  - Does not import `transition_engines`.
  - Does not directly assign to `GlobalState` fields.
- `tests/test_turn_loop_stub.py`
  - Added 14 focused M6 tests covering silent/no-action turns, diagnostic
    orders, diagnostic release-before-selection, treatment orders, nurse
    shadow execution, reactive bedside nurse speech, invalid action drops,
    verbal message commits, emotion call/skip behavior, `advance_turn`,
    active-only observations, no direct transition-engine import, and both
    termination paths.
- `docs/milestones/M6_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code
- existing scenario/evaluation output files

Existing production modules were not rewritten. M6 only added new turn-loop,
stub-agent, and termination modules plus focused tests.

## Tests Run and Results

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
14 passed in 0.06s
```

```bash
python -m pytest tests/test_orchestrator.py tests/test_observation_builder.py tests/test_action_validator.py tests/test_turn_loop_stub.py
```

Result: passed.

```text
92 passed in 0.11s
```

```bash
python -c "import ed_world_model; import ed_world_model.orchestration.turn_loop; import ed_world_model.agents.stubs"
```

Result: passed.

The full `tests/test_state_manager.py` suite was not run in this session because
the M5 handoff documents unrelated pre-existing `PatientEmotion.notes` failures
there. M6's focused and adjacent loop-facing suites passed.

## Deviations From `docs/v1_3_1.md`

- No intentional behavioral deviation.
- `TurnResult` is intentionally compact and test-oriented, not a full
  trajectory logger.
- Pending-question creation/resolution is hook-based only. The current
  `StateManager` does not expose pending-question mutate methods, so the turn
  loop calls them only if a later `StateManager` provides them. It does not
  mutate pending-question state directly.
- The reactive nurse bedside slot receives a small bedside observation rather
  than a new full observation schema. It is not selected by the Orchestrator.

## Known Limitations

- No real LLM agents, prompts, external API calls, or OpenAI client usage.
- No scenario loader.
- No full `TrajectoryLogger`.
- No nurse physical action.
- No patient/relative behavior actions.
- No relative emotion transition.
- No memory summarization.
- No task queue, async workflow, retries, plugin framework, PK/PD timing, or
  engine routing.
- Patient `can_speak` gating remains observation-driven; M6 does not add a new
  non-clinician verbal validator.
- The turn loop expects an injected physiology adapter with
  `predict(global_state, engine_facing_action)`.

## Exact Next Step

M7 Clinician LLM proposal parser.

## M6_patch

- `target` was intentionally removed from `VerbalAction` and was not added
  back.
- `VerbalAction.message_recipient` now returns `recipient` only.
- Kept `extra="forbid"` behavior on `VerbalAction`.
- Corrected the stale M6 test status: the previous `14 passed` status was
  wrong after `target` was removed because `message_recipient` still referenced
  the deleted field.
- `nurse_bedside_verbal_slot.payload["spoke"]` now derives from actual
  reactive bedside-slot committed messages.
- `tests/test_turn_loop_stub.py` now asserts that a reactive nurse bedside
  verbal message records `spoke is True`.
- When the nurse is active at turn start in the same turn as a valid treatment
  order and returns a verbal action, that nurse verbal is deferred into the
  reactive bedside slot so the corresponding `nurse_bedside_verbal_slot` event
  records `spoke=True`.
- Added coverage that a triggered bedside slot with a silent nurse records
  `spoke=False`.
- `StateManager.apply_physiology_update(...)` now merges partial physiology
  updates into existing `vitals`, `features`, and `status_flags` instead of
  replacing whole sub-objects.
- Missing physiology fields preserve existing state.
- Explicitly provided `None` values still overwrite existing values.
- Added StateManager coverage for partial feature, vital, and status-flag
  merges; explicit `None`; non-physiology state preservation; and existing
  physiology event behavior.
- Added turn-loop coverage that treatment physiology output with partial
  features does not drop existing feature keys.

Patch test run:

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
16 passed in 0.09s
```

Optional adjacent patch run:

```bash
python -m pytest tests/test_orchestrator.py tests/test_observation_builder.py tests/test_turn_loop_stub.py
```

Result: passed.

```text
63 passed in 0.09s
```

Manual verification:

```bash
python manual_test.py
```

Result: Turn 1 now records
`nurse_bedside_verbal_slot.payload["spoke"] == True` when the nurse message is
committed in the treatment turn.

StateManager physiology merge patch run:

```bash
python -m pytest tests/test_state_manager.py tests/test_turn_loop_stub.py
```

Result: failed only in the two pre-existing `PatientEmotion.notes` tests
unrelated to this patch; all M6 turn-loop tests passed.

```text
2 failed, 49 passed in 0.16s
```

Patch-specific merge run:

```bash
python -m pytest tests/test_state_manager.py tests/test_turn_loop_stub.py -k "physiology_update or partial_physiology or partial_feature"
```

Result: passed.

```text
10 passed, 41 deselected in 0.06s
```

Turn-loop focused rerun:

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
17 passed in 0.06s
```

Exact next step remains M7 Clinician LLM proposal parser.
