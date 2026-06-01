# M14a Dedup Patch Handoff

## Scope

This patch fixes duplicate accumulation for `known_history`,
`known_allergies`, and `known_medications` after structured
`FactExtractionResult` application.

No ontology, synonym mapping, semantic matching, LLM extraction, transition
engine change, scenario JSON change, or hidden-truth copying was added.

## Files Modified

- `ed_world_model/state/state_manager.py`
  - Replaced non-symptom append/dedup behavior with merge-by-identity behavior.
  - `source_texts` are no longer part of identity keys.
  - Matching history/allergy/medication facts now append new `source_texts`
    without exact duplicates.
  - Different status remains separate.
  - Different allergy reaction remains separate unless it matches after basic
    normalization.
- `tests/test_state_manager.py`
  - Added focused coverage for repeated history/allergy/medication disclosures.
  - Added coverage that exact duplicate `source_texts` are not duplicated.
  - Kept coverage that symptoms still merge by normalized name.
  - Kept coverage that no synonym matching is introduced.

## Merge Rules

- History identity:
  - normalized `item`
  - `status`
- Allergy identity:
  - normalized `substance`
  - `status`
  - normalized `reaction`
- Medication identity:
  - normalized `name`
  - `status`

Normalization remains basic only:

- strip
- lowercase
- collapse whitespace
- replace hyphens with spaces

`source_texts` are audit evidence, not identity keys.

## Tests Run

```bash
python -m pytest tests/test_state_manager.py tests/test_fact_extractor.py tests/test_turn_loop_stub.py
```

Result: passed.

```text
92 passed in 0.26s
```

```bash
python -m pytest tests/test_global_state.py tests/test_observation_builder.py tests/test_trajectory_logger.py
```

Result: passed.

```text
67 passed in 0.17s
```

## Notes

- Raw dialogue remains only in `runtime_state.messages` and trajectory
  committed message fields.
- Symptom merge behavior was not changed beyond using the existing shared
  source-text append helper.
- No files under `transition_engines/`, `transitions/`, `pdf/`, or `out/` were
  modified.
