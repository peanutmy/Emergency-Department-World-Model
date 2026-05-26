# M11 Turn Phase Patch Handoff

## Files Modified

- `ed_world_model/orchestration/turn_loop.py`
  - Changed turn execution from all-agent precommit generation to phased
    generation.
  - Non-clinician active agents now generate and commit first so required
    patient/relative responses and nurse result reports become visible before
    clinician generation.
  - Clinician observation is rebuilt after those committed messages.
  - Clinician-generated questions remain pending for a later turn instead of
    being answerable in the same turn.
  - Nurse bedside verbal slot remains reactive after valid treatment orders and
    is separate from turn-start nurse reporting.
- `ed_world_model/state/state_manager.py`
  - Added optional `created_before_turn` filtering to
    `resolve_pending_questions_for_agent(...)`.
  - Turn loop uses this filter to avoid resolving same-turn questions.
- `tests/test_turn_loop_stub.py`
  - Updated old parallel-order expectations to response-before-clinician
    ordering.
  - Added coverage that patient answers and nurse reports are committed before
    clinician generation and appear in clinician observation.
  - Added coverage that a clinician question created in the current turn remains
    unresolved for a later patient response.
- `tests/test_state_manager.py`
  - Added focused coverage for skipping same-turn pending questions during
    resolution.

No Orchestrator, ActionValidator, transition engine, scenario JSON, or agent
prompt files were modified by this patch.

## Tests Run and Results

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_full_turn_loop_integration.py tests/test_demo_runner.py tests/test_state_manager.py::test_resolve_pending_questions_can_skip_same_turn_questions tests/test_clinician_agent.py
```

Result: passed.

```text
93 passed in 0.31s
```

Broader run attempted:

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_state_manager.py tests/test_full_turn_loop_integration.py tests/test_demo_runner.py
```

Result: failed in two existing `tests/test_state_manager.py` patient-emotion
tests because `PatientEmotion.notes` is forbidden by the current schema.

```text
2 failed, 91 passed in 0.51s
```

The failures were:

- `test_update_patient_emotion_updates_patient_emotion_only`
- `test_partial_update_patient_emotion_preserves_prior_fields`

## Worktree Note

The worktree also had unrelated modifications in:

- `ed_world_model/actions/registry.py`
- `transitions/Cardiology/Acute Respiratory Distress.json`

Those files were not modified by this patch.

## Exact Next Step

Run a real trajectory with real agents and Hybrid physiology to confirm patient
answers and nurse result reports appear before clinician reasoning in the same
turn, then inspect whether focused history questions occur earlier without
blocking immediate ABC stabilization.
