# M2b Handoff

## Files Created or Modified

- `ed_world_model/actions/validator.py`
  - Added lightweight `ValidationResult`.
  - Added `ActionValidator` with dict-based validation for clinician proposals.
  - Added clinician no-action handling for `None`, `{"type": None}`, and
    proposal-level null action values.
  - Added one-action-per-turn enforcement for clinician proposals.
  - Added medical treatment validation for action type, family, `kind_hint`,
    family/`kind_hint` match, clinician-selectable status, params shape, extra
    params, and `raw_text` rejection.
  - Added params normalization from `ActionRegistry` templates. Missing params
    are filled with template defaults, including `None` values.
  - Added diagnostic order validation against `global_state.truth_state.test_bank`
    names only.
  - Medical treatment normalization produces an engine-facing action with
    `raw_text: None`; null clinician action does not become `no_action`.
- `tests/test_action_validator.py`
  - Added 22 focused tests covering requested M2b behavior and boundaries.
- `docs/milestones/M2b_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `ed_world_model/actions/registry.py`
- `ed_world_model/actions/__init__.py`
- `ed_world_model/state/state_manager.py`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

No Orchestrator, ObservationBuilder, adapters, agents, turn loop, nurse shadow
execution, physiology calls, emotion calls, or `GlobalState` mutation were
implemented.

## Tests Run and Results

```bash
python -m pytest tests/test_action_validator.py
```

Result: passed.

```text
22 passed in 0.06s
```

```bash
python -m pytest tests/test_action_registry.py tests/test_action_validator.py
```

Result: passed.

```text
45 passed in 0.10s
```

Broad pytest was not run, per M2b focused-test scope.

## M2b_patch

- Removed the dead `action_type is None` branch from
  `validate_clinician_proposal`; explicit null actions are handled by
  `_is_null_action`.
- Tightened `_is_null_action` so only `None` and explicit `{"type": None}`
  action objects are treated as null actions.
- Rejected `{"action": [...]}` list-form proposals with a validation error.
  During the v1.3.1 dict-based phase, `ActionValidator` still accepts several
  dict shapes beyond the literal section 8 sketch, including direct action
  dicts, canonical `{"action": {...}}` wrappers, and legacy
  `medical_treatment_order` / `diagnostic_order` wrapper keys.
- Kept conservative recursive `raw_text` rejection: any `raw_text` key anywhere
  in a clinician proposal fails validation.
- Removed unused `ValidationResult.warnings`.
- Added focused tests for nested `raw_text`, verbal-action `raw_text`,
  non-dict proposals, canonical action wrappers, list-form rejection,
  diagnostic raw_text, missing/empty diagnostic `test_name`, and frozen
  `ValidationResult`.

Patch test runs:

```bash
python -m pytest tests/test_action_validator.py
```

Result: passed.

```text
31 passed in 0.08s
```

```bash
python -m pytest tests/test_action_registry.py tests/test_action_validator.py
```

Result: passed.

```text
54 passed in 0.08s
```

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- `docs/v1_3_1.md` includes an illustrative treatment example with
  `family: "oxygen_support"`. M2b follows the canonical M2a registry family
  names, so oxygen support validates under `family: "respiratory_support"` and
  `kind_hint: "oxygen_support"`.
- M2b uses dict-based validation because no LLM proposal schema models exist
  yet. Pydantic proposal schemas remain deferred.

## Known Limitations

- `ActionValidator` validates proposals only; it does not execute actions.
- It does not judge clinical correctness or infer hidden diagnosis.
- It does not infer, release, or expose diagnostic results.
- It does not create `PendingDiagnosticResult`; `StateManager` remains
  responsible for diagnostic creation and release.
- It does not create nurse shadow execution events.
- It does not create `no_action` for physiology. Later turn-loop/adapter code
  must create `{"raw_text": None, "kind_hint": "no_action", "params":
  {"elapsed_min": 1}}` when no valid treatment order exists.
- It does not validate patient, nurse, or relative verbal content.
- `param_options` remain guidance only and are not enforced as value constraints.

## Exact Next Step

M4a Orchestrator.

Diagnostic pending-result creation and ready-result release are already covered
by `StateManager` from M1b, including test-bank `turnaround_turns` fallback
behavior and release into `known_facts.available_results`.
