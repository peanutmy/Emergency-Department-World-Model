# M14b Small Patch Handoff

## Scope

This patch applies two hardening fixes before real-agent +
`LLMFactExtractor` trajectory runs.

No FactExtractor redesign, ontology/synonym mapping, source-text substring
guard, transition-engine change, or scenario JSON change was added.

## Changes

- `ed_world_model/facts/llm_extractor.py`
  - Removed `available_results` from the known-facts context whitelist used in
    the LLMFactExtractor prompt.
  - `available_results` remains in runtime `known_facts` and trajectory
    snapshots, but is no longer supplied to the disclosure extractor as an
    anchor.
  - Added redaction of `sk-...` API-key-like substrings when wrapping LLM client
    exceptions in `FactExtractionError`.
- `tests/test_fact_extractor.py`
  - Added prompt coverage that `available_results` and released-result text do
    not appear in the LLMFactExtractor prompt.
- `tests/test_trajectory_logger.py`
  - Added regression coverage that a client failure containing
    `sk-test-secret` is redacted in `fact_extraction_error` event payloads and
    in serialized trajectory JSON.

## Behavior Notes

- The extractor still receives disclosure-relevant context:
  - `chief_complaint`
  - `known_symptoms`
  - `known_history`
  - `known_allergies`
  - `known_medications`
- Released diagnostic results are intentionally excluded from extractor prompt
  context to reduce pressure to infer symptoms from test data.
- Extractor errors still do not crash the turn; the turn loop records
  `fact_extraction_error` and continues.

## Deferred

- Source-text grounding/substr matching remains deferred.
- The `PatientEmotion` schema issue observed earlier remains a separate
  follow-up.

## Tests Run

```bash
python -m pytest tests/test_fact_extractor.py tests/test_turn_loop_stub.py tests/test_trajectory_logger.py
```

Result: passed.

```text
83 passed in 0.31s
```

Verification:

```bash
git diff -- transition_engines transitions
```

Result: no diff.
