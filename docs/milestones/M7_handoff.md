# M7 Handoff

## Files Created or Modified

- `ed_world_model/scenario_loader.py`
  - Added `ScenarioLoader` for converting an existing extracted scenario JSON
    file or dictionary into a v1.3.1 `GlobalState`.
  - Added `load_scenario(...)` convenience function.
  - Reads only scenario-level JSON fields:
    - `scenario_description`
    - `case_context.demographics`
    - `case_context.history`
    - `case_context.allergies`
    - `case_context.home_medications`
    - `case_context.chief_complaint`
    - `case_context.symptoms`
    - `case_context.disclosure_rules`
    - `case_context.supporting_findings`
    - `case_context.baseline_vitals`
    - `case_context.baseline_features`
  - Populates:
    - `truth_state.scenario_description`
    - `truth_state.demographics`
    - `truth_state.patient_internal_state`
    - `truth_state.test_bank`
    - `patient_state.vitals`
    - `patient_state.features`
    - `runtime_state.turn_index = 0`
  - Leaves `known_facts`, pending diagnostic results, messages, and events at
    their defaults.
  - Uses `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2` when a supporting finding
    does not provide a usable `turnaround_turns`.
  - Does not import or call `transition_engines`.
  - Ignores `pairs` entirely.
- `tests/test_scenario_loader.py`
  - Added 28 focused tests covering dict and file loading, hidden truth
    population, test-bank conversion, baseline vitals/features, missing
    baseline behavior, known-facts/runtime defaults, pair ignoring, input
    immutability, and no transition-engine import.
- `tests/fixtures/scenario_loader_minimal.json`
  - Added a small representative scenario-loader fixture.
- `docs/milestones/M7_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code
- existing test files
- clinician/patient/nurse/relative LLM agents
- turn-loop code

No PDFs are read, no LLMs are called, no new transition pairs are extracted,
and transition pairs are not used as runtime trajectory or baseline state.

## Tests Run and Results

```bash
python -m pytest tests/test_scenario_loader.py
```

Result: passed.

```text
28 passed in 0.06s
```

```bash
python -c "import ed_world_model; import ed_world_model.scenario_loader"
```

Result: passed.

Adjacent state check requested by the M7 prompt:

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_scenario_loader.py
```

Result: failed only in the pre-existing `PatientEmotion.notes` cases documented
in prior handoffs; all M7 scenario-loader tests passed in this run.

```text
3 failed, 79 passed in 0.14s
```

Failing tests:

- `tests/test_global_state.py::test_patient_emotion_remains_separate_from_patient_profile`
- `tests/test_state_manager.py::test_update_patient_emotion_updates_patient_emotion_only`
- `tests/test_state_manager.py::test_partial_update_patient_emotion_preserves_prior_fields`

All three failures reject extra `notes` on `PatientEmotion` and are unrelated
to `ScenarioLoader`.

## Deviations From `docs/v1_3_1.md`

- No intentional behavioral deviation.
- `ScenarioLoader` is an initialization boundary that constructs a fresh
  `GlobalState`; it does not mutate an existing runtime state.
- The loader has a small optional `strict=True` mode. Default mode tolerates
  missing `baseline_vitals` and `baseline_features` by leaving vitals/features
  at model defaults, as requested.
- Unknown demographic keys are ignored because `Demographics` has a strict
  schema and only supports `name`, `age`, `sex`, and `weight_kg`.

## Known Limitations

- The loader only maps current v1.3.1 fields and does not validate the full
  extracted-scenario JSON schema.
- It does not load agent profiles from scenario JSON.
- It does not infer chief complaint or symptoms from legacy top-level
  `chief_complaint` / `Symptoms`; it reads only
  `case_context.chief_complaint` and `case_context.symptoms`.
- It does not deduplicate test-bank entries.
- It does not create pending diagnostics, release results, build observations,
  call adapters, or run a turn.
- It intentionally ignores `pairs`, including `pairs[0].input.before`, even
  when scenario-level baseline vitals/features are missing.

## Exact Next Step

Clinician LLM proposal parser.
