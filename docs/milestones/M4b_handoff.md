# M4b Handoff

## Files Created or Modified

- `ed_world_model/orchestration/observation_builder.py`
  - Added `ObservationBuilder`.
  - Added `build_for(active_agents, global_state)` for active-agent-only
    observation construction.
  - Added role-specific builders:
    - `build_clinician_observation(global_state)`
    - `build_nurse_observation(global_state)`
    - `build_patient_observation(global_state)`
    - `build_relative_observation(global_state)`
  - Reused canonical M4a agent constants from
    `ed_world_model.orchestration.orchestrator`.
  - Uses simple dict observations because no dedicated observation Pydantic
    models exist yet.
  - Reads from `GlobalState` only and does not mutate state.
  - Exposes diagnostic test names from `truth_state.test_bank`, but not
    unreleased test results.
  - Includes released per-turn diagnostic results through
    `runtime_state.newly_available_results`.
  - Uses `runtime_state.messages` for recent message windows and
    `runtime_state.last_turn_events` for previous-turn system events.
  - Sanitizes observation payloads by removing any `raw_text` keys recursively.
- `tests/test_observation_builder.py`
  - Added 20 focused tests covering active-agent filtering, role-specific
    fields, hidden diagnostic and patient-internal truth boundaries,
    read-only behavior, absence of transition-engine and state-manager
    coupling, previous-turn event visibility, per-turn result visibility,
    missing optional defaults, and `raw_text` exclusion.
  - M4b approved follow-up added 2 chief-complaint visibility tests: clinician
    sees chief complaint only after it is in `known_facts`, and hidden
    `patient_internal_state.chief_complaint` is not leaked.
- `ed_world_model/state/global_state.py`
  - M4b approved follow-up added `KnownFacts.chief_complaint: str | None` as a
    released clinical field.
- `ed_world_model/state/state_manager.py`
  - M4b approved follow-up added `chief_complaint` support to
    `StateManager.update_known_facts(...)`.
- `tests/test_known_facts_chief_complaint.py`
  - M4b approved follow-up added focused coverage for the optional known-facts
    field and the StateManager mutation boundary.
- `docs/milestones/M4b_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `ed_world_model/orchestration/orchestrator.py`
- `ed_world_model/actions/registry.py`
- `ed_world_model/actions/validator.py`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing engine code
- existing test files

The worktree already had unrelated modifications to
`ed_world_model/state/global_state.py` and
`transitions/Cardiology/Acute Respiratory Distress.json`. M4b approved
follow-up edited only the `KnownFacts` section of `global_state.py` and did not
edit or revert the scenario JSON file.

## Tests Run and Results

```bash
python -m pytest tests/test_observation_builder.py
```

Result: passed.

```text
20 passed in 0.10s
```

```bash
python -m pytest tests/test_orchestrator.py tests/test_observation_builder.py
```

Result: passed.

```text
35 passed in 0.06s
```

M4b approved chief-complaint follow-up:

```bash
python -m pytest tests/test_known_facts_chief_complaint.py tests/test_observation_builder.py
```

Result: passed.

```text
24 passed in 0.11s
```

```bash
python -m pytest tests/test_orchestrator.py tests/test_observation_builder.py tests/test_known_facts_chief_complaint.py
```

Result: passed.

```text
39 passed in 0.08s
```

```bash
python -m pytest tests/test_state_manager.py::test_update_known_facts_updates_known_facts_without_touching_truth_state
```

Result: passed.

```text
1 passed in 0.06s
```

Broad pytest was not run because M4b only adds ObservationBuilder behavior and
the requested focused plus adjacent Orchestrator suites passed. Full
StateManager and GlobalState suites were not run because the worktree already
contained unrelated pending `PatientEmotion` edits.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- The approved chief-complaint follow-up adds a `known_facts.chief_complaint`
  field even though `docs/v1_3_1.md` lists only the original known-facts
  subfields. This follows the existing architecture rule that clinical-team
  factual knowledge should be released through `known_facts`, not read directly
  from hidden `patient_internal_state`.
- Observation outputs are simple dictionaries instead of Pydantic observation
  models because no suitable observation models exist yet.
- Visible bedside event filtering uses the currently available generic
  `Event.type` / `Event.payload["visible_to"]` shape because no dedicated
  bedside-event model exists yet.

## Known Limitations

- ObservationBuilder is construction-only; it does not decide active agents,
  validate actions, release diagnostics, call adapters, call engines, call
  agents, or run the turn loop.
- There are no dedicated typed observation schemas yet.
- Recent-message filtering relies on the existing `Message.recipient` field.
  Clinician and nurse receive the simple recent message window; patient and
  relative receive public, addressed, or self-spoken messages.
- Relative family-side hidden knowledge is represented as
  `family_side_hidden_info: None` because v1.3.1 has no family-side knowledge
  model yet.
- Bedside-event visibility is conservative and heuristic until later turn-loop
  code defines exact event types.

## Exact Next Step

M5 Engine adapters.

## M4b_patch

### Files Modified

- `ed_world_model/orchestration/observation_builder.py`
  - Patient observations now include patient-owned hidden internal facts:
    `hidden_history`, `hidden_allergies`, and `hidden_home_medications`, along
    with the existing `chief_complaint`, `symptoms`, and `disclosure_rules`.
  - Clinician observations no longer include `safe_case_context` or raw
    `truth_state.scenario_description`.
  - Clinician pending-question visibility now includes only unresolved,
    required-response questions where the clinician is the source or target.
    This uses the existing `PendingQuestion.source_agent` and `target_agent`
    fields.
  - Nurse observation field `current_bedside_event_if_any` was renamed to
    `last_bedside_event_if_any` because it is derived from
    `runtime_state.last_turn_events`, not `current_turn_events`.
  - Relative `visible_patient_status` is now coarse: it includes status flags
    and selected visible feature keys only, not full numeric vitals.
  - Observations continue to ignore `runtime_state.current_turn_events`.
- `tests/test_observation_builder.py`
  - Added and updated focused tests for patient hidden fields, clinician raw
    scenario-description exclusion, clinical-team hidden-truth boundaries,
    patient known-facts exclusion, coarse relative status, private-message
    filtering, returned-observation copy safety, clinician-scoped pending
    questions, current-turn event exclusion, last-turn event inclusion,
    `last_bedside_event_if_any`, and `raw_text` stripping.
- `docs/milestones/M4b_handoff.md`
  - Added this patch section.

### Files Intentionally Not Touched

- `ed_world_model/orchestration/orchestrator.py`
- `ed_world_model/state/state_manager.py`
- `ed_world_model/actions/registry.py`
- `ed_world_model/actions/validator.py`
- `transition_engines/`
- `transitions/`
- scenario JSON data
- `pdf/`
- `out/`

### Tests Run and Results

```bash
python -m pytest tests/test_observation_builder.py
```

Result: passed.

```text
32 passed in 0.08s
```

```bash
python -m pytest tests/test_orchestrator.py tests/test_observation_builder.py
```

Result: passed.

```text
47 passed in 0.07s
```

### Known Limitations

- Relative visible feature selection is conservative and key-based until later
  turn-loop/bedside-event modeling defines a richer visible status contract.
- Observation outputs remain simple dictionaries.

### Exact Next Step

M5 Engine adapters.
