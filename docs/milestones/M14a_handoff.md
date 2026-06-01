# M14a Handoff

## Scope

M14a adds the FactExtractor / DisclosureMapper foundation for structured
patient and relative verbal disclosures.

This milestone does not add clinical reasoning, diagnosis extraction, ontology
mapping, synonym mapping, real LLM extraction, or hidden-truth copying.

## Files Created

- `ed_world_model/facts/__init__.py`
  - Exposes fact schemas and extractor interfaces.
- `ed_world_model/facts/schemas.py`
  - Added `KnownSymptom`, `KnownHistory`, `KnownAllergy`,
    `KnownMedication`, and `FactExtractionResult`.
- `ed_world_model/facts/extractor.py`
  - Added `FactExtractionContext`, `FactExtractor`, `NoopFactExtractor`, and
    `FakeFactExtractor`.
- `tests/test_fact_extractor.py`
  - Added focused schema and fake/noop extractor tests.
- `docs/milestones/M14a_handoff.md`
  - Added this handoff.

## Files Modified

- `ed_world_model/state/global_state.py`
  - Upgraded existing `KnownFacts` fields:
    - `known_symptoms: list[KnownSymptom]`
    - `known_history: list[KnownHistory]`
    - `known_allergies: list[KnownAllergy]`
    - `known_medications: list[KnownMedication]`
  - Did not add `structured_*` duplicate fields.
  - Did not add `known_patient_statements` or `known_relative_statements`.
- `ed_world_model/state/state_manager.py`
  - Added `apply_fact_extraction_result(...)`.
  - Added symptom upsert/merge by normalized symptom name.
  - Removed raw string fact extension behavior from known-fact updates.
  - Keeps history/allergy/medication merging simple, with exact duplicate
    prevention.
- `ed_world_model/orchestration/turn_loop.py`
  - Added optional `fact_extractor` injection.
  - Patient and relative committed verbal actions now call the extractor only
    when one is provided.
  - Clinician and nurse speech do not call the extractor.
  - Extraction errors are recorded as `fact_extraction_error` events and do not
    crash the turn.
  - Applied extraction results record `fact_extraction_applied` events with
    counts only, not raw dialogue.
- `ed_world_model/orchestration/runner.py`
  - Added optional `fact_extractor` dependency injection for integration runs.
- Focused tests updated in:
  - `tests/test_global_state.py`
  - `tests/test_state_manager.py`
  - `tests/test_turn_loop_stub.py`
  - `tests/test_observation_builder.py`
  - `tests/test_trajectory_logger.py`
  - `tests/test_hybrid_physiology_adapter.py`

## Schema Notes

- `KnownSymptom` fields:
  - `name`
  - `status`
  - `onset`
  - `severity`
  - `source_texts`
- `KnownSymptom` intentionally has no `course`, `notes`, ontology fields, or
  synonym fields.
- `KnownHistory`, `KnownAllergy`, and `KnownMedication` intentionally have no
  `notes`.
- `source_texts` are stored on structured facts for audit.
- Raw patient/relative statement lists are not stored in `known_facts`.

## Runtime Behavior

- Raw dialogue remains in:
  - `runtime_state.messages`
  - trajectory `committed_messages` / demo `messages`
- `known_facts` now stores extracted structured facts only.
- If no extractor is provided, patient/relative messages are still committed,
  but `known_facts` is not updated from raw text.
- If extraction returns `ignored=True`, `known_facts` is not updated.
- If extraction raises, a `fact_extraction_error` event is recorded and the
  turn continues.
- The extractor context includes recent messages, current `known_facts`, and
  the last pending question for that speaker. It does not receive
  `truth_state` or `patient_internal_state`.

## Symptom Merge Rules Implemented

- Symptoms are matched only by normalized name:
  - strip
  - lowercase
  - collapse whitespace
  - replace hyphens with spaces
- No ontology or synonym matching was added.
- `SOB` and `shortness of breath` remain separate if the extractor emits those
  different names.
- Existing `onset` and `severity` are filled only when currently `None`.
- `source_texts` append without exact duplicates.
- `uncertain` status can be updated to `present` or `absent`.
- Conflicting non-uncertain statuses are not overwritten in M14a.

## Tests Run

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_turn_loop_stub.py tests/test_observation_builder.py tests/test_trajectory_logger.py
```

Result: passed.

```text
141 passed in 0.22s
```

```bash
python -m pytest tests/test_fact_extractor.py
```

Result: passed.

```text
11 passed in 0.05s
```

Final combined focused run:

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_turn_loop_stub.py tests/test_observation_builder.py tests/test_trajectory_logger.py tests/test_fact_extractor.py
```

Result: passed.

```text
152 passed in 0.16s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_demo_runner.py
```

Result: passed.

```text
43 passed in 0.47s
```

```bash
python -m pytest tests/test_hybrid_physiology_adapter.py
```

Result: passed.

```text
9 passed in 0.08s
```

```bash
python -m pytest tests/test_known_facts_chief_complaint.py tests/test_scenario_loader.py
```

Result: passed.

```text
30 passed in 0.09s
```

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
32 passed in 0.06s
```

Adjacent reruns after final hardening:

```bash
python -m pytest tests/test_scenario_loader.py
```

Result: passed.

```text
28 passed in 0.07s
```

```bash
python -m pytest tests/test_hybrid_physiology_adapter.py
```

Result: passed.

```text
9 passed in 0.06s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_demo_runner.py tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
75 passed in 0.32s
```

One broad adjacent combined run that placed
`tests/test_hybrid_physiology_adapter.py` before `tests/test_scenario_loader.py`
failed in `test_loader_does_not_import_transition_engines` because the Hybrid
adapter test imports `transition_engines` and the scenario-loader test asserts
that module is absent from `sys.modules`. The same suites passed when rerun in
separate pytest processes.

```bash
python -c "import ed_world_model; import ed_world_model.facts; import ed_world_model.orchestration.runner"
```

Result: passed.

## Notes

- A stale test expectation that supplied `PatientEmotion.notes` was removed
  from touched tests to match the existing strict `PatientEmotion` schema.
- No files under `transition_engines/`, `transitions/`, `pdf/`, or `out/`
  were modified.
- Real LLM fact extraction remains deferred.
