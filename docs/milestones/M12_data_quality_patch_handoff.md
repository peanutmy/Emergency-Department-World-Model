# M12 Data Quality Patch Handoff

## Scope

- `known_facts` now updates from committed patient and relative verbal
  disclosures.
  - Patient verbal content is appended to `known_facts.known_symptoms`.
  - Relative verbal content is appended to `known_facts.known_history`.
  - Exact duplicate strings are not appended again.
  - Clinician and nurse verbal actions do not update `known_facts`.
  - Hidden `truth_state` is not copied into `known_facts`.
- Patient verbal actions are not committed when the patient is not alive, not
  conscious, or cannot speak according to `patient_state.status_flags`.
- Medication-like medical treatment actions now require a non-empty
  `params.drug_name`.
  - `dose` and `unit` may remain null.
  - `oxygen_support` and `airway_management` are not affected.
- Clinician prompt guidance now tells the model to specify `drug_name` for
  medication-like actions and to avoid inventing unsupported drug names.
- Trajectory clinician actions no longer duplicate `normalized_action`.
  - Medical treatment clinician actions keep `type`, `action_type`, `family`,
    `kind_hint`, and `params`.
  - Diagnostic clinician actions keep `type`, `action_type`, and `test_name`.
  - `physiology_action` remains separate with `raw_text`, `kind_hint`, and
    `params`.

No clinical hard rules were added. `ActionValidator` only performs structural
input-quality validation for medication-like `drug_name`.

## Files Changed

- `ed_world_model/orchestration/turn_loop.py`
  - Added `update_known_facts_from_verbal_action(...)`.
  - Calls the helper only after a verbal action is committed.
  - Blocks patient speech when status flags indicate the patient cannot speak.
- `ed_world_model/actions/validator.py`
  - Rejects medication-like treatment orders with missing, null, or blank
    `params.drug_name`.
- `ed_world_model/actions/registry.py`
  - Updated medication-like `drug_name` prompt guidance.
- `ed_world_model/agents/clinician.py`
  - Added clinician prompt guidance for medication-like `drug_name`.
- `ed_world_model/demo.py`
  - Removed `normalized_action` from demo `clinician_action` serialization.
- `ed_world_model/trajectory/logger.py`
  - Normalizes `clinician_action` output without `normalized_action`.
- Focused tests updated in:
  - `tests/test_action_validator.py`
  - `tests/test_clinician_agent.py`
  - `tests/test_turn_loop_stub.py`
  - `tests/test_demo_runner.py`
  - `tests/test_trajectory_logger.py`

No files under `transition_engines/` or `transitions/` were modified.

## Tests Run

```bash
python -m pytest tests/test_action_validator.py tests/test_clinician_agent.py tests/test_turn_loop_stub.py tests/test_demo_runner.py tests/test_trajectory_logger.py
```

Result: passed.

```text
153 passed in 0.37s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
23 passed in 0.24s
```

```bash
git diff -- transition_engines transitions
```

Result: no diff.

## Patient/Verbal-Agent Patch

- Patient prompt now explicitly forbids invented patient facts:
  symptoms, history, allergies, medications, social history, and
  review-of-systems findings.
- Patient prompt now says to disclose only information present in the
  observation, `patient_internal_state`, `disclosure_rules`, or recent
  conversation.
- Patient prompt now forbids plausible disease-associated symptom additions
  when they are not shown, and directs conservative answers for unknown or
  unlisted facts.
- Verbal-only patient, nurse, and relative prompts now restrict
  `requires_response=true` to explicit questions requiring the target agent to
  answer.
- Verbal-only parser normalization now converts over-broad
  `requires_response=true` to `false` when content is not an explicit targeted
  question.
- `known_facts` verbal updates remain shallow disclosure capture, not semantic
  extraction. Patient disclosures may append to `known_symptoms`; relative
  disclosures may append to `known_history`; clinician and nurse verbal actions
  do not update patient facts.
- Normalized exact-string dedup was added for shallow patient/relative
  disclosure capture. Different wording with similar meaning can still be
  stored.
- No clinical hard rules, family-level rules, kind_hint-level rules, medical
  correctness validation, LLM fact extraction, semantic deduplication, behavior
  actions, nurse physical actions, task queues, new memory systems, scenario
  JSON changes, transition-engine changes, or physiology behavior changes were
  added.

## Patient/Verbal-Agent Patch Tests

```bash
python -m pytest tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py tests/test_turn_loop_stub.py
```

Result: passed.

```text
119 passed in 0.14s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_demo_runner.py
```

Result: passed.

```text
42 passed in 0.30s
```

```bash
git diff -- transition_engines transitions
```

Result: no diff.
