# M14b Handoff

## Scope

M14b adds an opt-in real LLM-backed `FactExtractor` for patient and relative
verbal disclosures.

The extractor maps explicitly spoken patient/relative facts into existing
structured `KnownFacts` schemas. It does not add clinical guidance rules,
diagnosis extraction, ontology mapping, synonym mapping, hidden-truth copying,
or ActionValidator clinical correctness checks.

## Files Created

- `ed_world_model/facts/llm_extractor.py`
  - Added `LLMFactExtractor`.
  - Builds a strict extraction prompt from safe context only.
  - Calls `llm_client.generate(prompt)` or a callable client.
  - Parses strict JSON into `FactExtractionResult`.
  - Raises `FactExtractionError` on invalid JSON, forbidden fields, schema
    mismatch, non-string response, or client failure.

## Files Modified

- `ed_world_model/facts/extractor.py`
  - Added shared `FactExtractionError`.
- `ed_world_model/facts/__init__.py`
  - Exposes `FactExtractionError` and `LLMFactExtractor`.
- `ed_world_model/demo.py`
  - Added `--fact-extractor none|fake|llm`.
  - Default is `none`.
  - `fake` constructs the existing deterministic fake extractor with an
    ignored default result.
  - `llm` constructs `LLMFactExtractor` with the existing OpenAI-compatible
    real client unless a factory is injected.
- `ed_world_model/agents/llm_client.py`
  - Updated the missing-key message to include `--fact-extractor llm`.
- `ed_world_model/trajectory/logger.py`
  - Carries `fact_extractor_mode` through trajectory and summary payloads.
- Focused tests updated in:
  - `tests/test_fact_extractor.py`
  - `tests/test_turn_loop_stub.py`
  - `tests/test_trajectory_logger.py`
  - `tests/test_demo_runner.py`

## Runtime Behavior

- Real LLM fact extraction is opt-in only:

```bash
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5 \
  --fact-extractor llm
```

- Default mode remains:

```bash
--fact-extractor none
```

- If `--fact-extractor llm` is selected without API/client config, construction
  fails clearly before any turn runs.
- Tests use fake clients only and do not require `OPENAI_API_KEY`.
- `NoopFactExtractor` and `FakeFactExtractor` remain available.

## Prompt Rules

The LLM prompt instructs the model to:

- extract only facts explicitly stated by the current patient/relative speaker
- preserve negation
- use `status="absent"` for denied symptoms
- use `status="uncertain"` for uncertainty
- avoid diagnosis and differential diagnosis
- avoid hidden facts and plausible medical inferences
- avoid ontology and synonym mapping
- avoid notes and course fields
- use recent messages, `last_question`, and existing `known_facts` only to
  resolve elliptical references
- return `ignored=true` when there is no extractable fact or the referent is
  unclear
- return JSON only

## Context Boundary

`LLMFactExtractor` receives only:

- current speaker
- current utterance
- last question for that speaker, if any
- recent sanitized messages
- existing sanitized `known_facts`

It does not receive full `GlobalState`, `truth_state`,
`patient_internal_state`, hidden test-bank results, transition pairs, source
PDFs, chain-of-thought, or API keys.

## Error Handling

- Blank content returns `ignored=True`.
- Non-patient/non-relative speakers return `ignored=True`.
- Invalid model output raises `FactExtractionError`.
- The existing turn loop catches extractor errors, records a
  `fact_extraction_error` event, and continues the turn.
- Applied extraction results still go through
  `StateManager.apply_fact_extraction_result(...)` and existing M14a/M14a
  dedup merge rules.

## Tests Run

```bash
python -m pytest tests/test_fact_extractor.py
```

Result: passed.

```text
30 passed in 0.38s
```

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
37 passed in 0.28s
```

```bash
python -m pytest tests/test_trajectory_logger.py
python -m pytest tests/test_demo_runner.py
```

Result: passed.

```text
15 passed in 0.28s
20 passed in 0.34s
```

Focused combined run:

```bash
python -m pytest tests/test_fact_extractor.py tests/test_turn_loop_stub.py tests/test_trajectory_logger.py tests/test_demo_runner.py tests/test_full_turn_loop_integration.py tests/test_llm_client.py
```

Result: passed.

```text
132 passed in 0.55s
```

Import check:

```bash
python -c "import ed_world_model; import ed_world_model.facts; import ed_world_model.demo"
```

Result: passed.

Verification:

```bash
git diff -- transition_engines transitions
```

Result: no diff.

## Known Limitations

- No provider-side JSON schema/tool calling was added.
- No prompt repair or retry loop was added.
- No ontology, synonym, semantic matching, or diagnosis extraction was added.
- Extraction quality depends on the selected real model following the prompt.
- The demo `fake` fact extractor mode is intentionally ignored-by-default
  unless tests or callers inject scripted results.

## Next Step

Run an opt-in real-agent trajectory with `--fact-extractor llm` and inspect
whether patient/relative disclosures populate `known_facts` as intended without
inventing facts from clinician/nurse assumptions.
