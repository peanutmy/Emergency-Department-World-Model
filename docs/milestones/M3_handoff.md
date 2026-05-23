# M3 Handoff

## Files Created or Modified

- `tests/test_diagnostic_flow.py`
  - Added lightweight integration tests for diagnostic ordering and release
    using the existing `ActionValidator` and `StateManager` directly.
  - Verified valid diagnostic orders normalize to only `type` and `test_name`.
  - Verified unknown diagnostic `test_name` values are rejected.
  - Verified validated diagnostic orders can be passed to
    `StateManager.create_pending_diagnostic_result`.
  - Verified turnaround timing comes from `truth_state.test_bank`, including the
    `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2` fallback when
    `turnaround_turns` is `None`.
  - Verified ready-release behavior, durable `known_facts.available_results`,
    per-turn `runtime_state.newly_available_results`, hidden unreleased
    results, `advance_turn()` clearing behavior, multiple releases in one call,
    current-turn diagnostic release events, absence of `raw_text`, unchanged
    `truth_state.test_bank`, and no diagnostic-triggered nurse shadow,
    physiology, or emotion events.
- `docs/milestones/M3_handoff.md`
  - Added this handoff.

No production code was modified for M3.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `ed_world_model/state/state_manager.py`
- `ed_world_model/actions/validator.py`
- `ed_world_model/actions/registry.py`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

No `DiagnosticWorkflow` class, diagnostic timing policy, test bundles,
multi-test ordering, diagnostic categories, async task queue, result
interpretation engine, Orchestrator, ObservationBuilder, adapters, agents,
physiology engine calls, emotion engine calls, or turn loop code was added.

## Tests Run and Results

```bash
python -m pytest tests/test_diagnostic_flow.py
```

Result: passed.

```text
7 passed in 0.07s
```

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_action_registry.py tests/test_action_validator.py tests/test_diagnostic_flow.py
```

Result: passed.

```text
110 passed in 0.11s
```

Broad pytest was not run because M3 only adds focused diagnostic-flow
integration coverage and the requested focused plus adjacent suites passed.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- `docs/v1_3_1.md` does not specify the exact `diagnostic_release` event
  payload. M3 preserves the current M1b event design:
  `test_name`, `ordered_at_turn`, `ready_at_turn`, and `released_at_turn`.
  The result itself is stored in `known_facts.available_results` and
  `runtime_state.newly_available_results`; it is not duplicated into the event
  payload by current production code.

## Known Limitations

- M3 is test-only integration coverage. It adds no new runtime helper.
- Diagnostic pending-result creation still happens by calling
  `StateManager.create_pending_diagnostic_result(...)` with the validated
  normalized action's `test_name`.
- Duplicate diagnostic orders are not deduplicated.
- Duplicate `truth_state.test_bank` names are still resolved by the existing
  `StateManager` behavior, which uses the first matching test name.
- There is still no Orchestrator, ObservationBuilder, adapter layer, agent
  implementation, trajectory logger, physiology integration, emotion
  integration, or turn loop.

## Exact Next Step

M4a Orchestrator.

## M3_patch

### Files Modified

- `tests/test_diagnostic_flow.py`
  - Added review-gap coverage for zero-turnaround same-turn diagnostic release.
  - Added no-pending release coverage: returns `[]` and emits no
    `diagnostic_release` event.
  - Added same-turn idempotency coverage: a second release call after the first
    release adds no known result, no newly available result, and no release
    event.
  - Added pre-ready hidden-result coverage: the unreleased result string is not
    present in `known_facts.available_results`,
    `runtime_state.newly_available_results`, or current/last event payloads.
  - Replaced the brittle `"raw_text" not in str(model_dump)` assertion with
    structural checks over normalized action keys and event payload keys.
- `ed_world_model/actions/validator.py`
  - Simplified `validate_diagnostic_order(...)` so direct diagnostic-order
    validation checks only for a top-level `raw_text` key. The diagnostic order
    shape is flat.
  - Kept the recursive `raw_text` rejection in
    `validate_clinician_proposal(...)` and medical treatment validation.
- `docs/milestones/M3_handoff.md`
  - Added this M3_patch section.

No Orchestrator, ObservationBuilder, adapters, agents, physiology engine calls,
emotion engine calls, turn loop, `DiagnosticWorkflow` class, task queue, async
workflow, retry path, or policy object was added.

### Tests Run and Results

```bash
python -m pytest tests/test_diagnostic_flow.py tests/test_action_validator.py
```

Result: passed.

```text
42 passed in 0.13s
```

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_action_registry.py tests/test_action_validator.py tests/test_diagnostic_flow.py
```

Result: passed.

```text
114 passed in 0.12s
```

### Review Items Fixed

- `turnaround_turns=0` now has M3 integration coverage for same-turn release.
- Empty pending-diagnostic release is covered as a no-op with no release event.
- Repeated release at the same turn is covered as idempotent after the first
  release.
- Hidden unreleased result strings are checked structurally against known
  results, newly available results, and current/last event payloads.
- Brittle raw-text substring checks were replaced with structural key checks.
- Direct diagnostic-order validation now uses a flat top-level `raw_text`
  check. Existing `tests/test_action_validator.py` coverage still rejects
  diagnostic orders with top-level `raw_text`, and the full clinician proposal
  path still performs recursive `raw_text` rejection.
- `diagnostic_order_created` remains the existing `StateManager` event behavior
  from M1b. It is not a new diagnostic workflow layer.

### Review Items Intentionally Deferred

- Release ordering by `ready_at_turn` / `ordered_at_turn`.
- Duplicate diagnostic order deduplication.
- Duplicate `truth_state.test_bank` name policy.
- Defensive `diagnostic_release_skipped` path.
- Any new workflow class, including `DiagnosticWorkflow`.

### Exact Next Step

M4a Orchestrator.
