# M5 Handoff

## Files Created or Modified

- `ed_world_model/adapters/hybrid_physiology_adapter.py`
  - Added `HybridPhysiologyAdapter`.
  - Builds Hybrid engine input from v1.3.1 runtime `GlobalState` fields and a
    validated engine-facing action.
  - Calls the existing `transition_engines.hybrid_engine.HybridEngine` when
    constructed with an `llm_client`, or a supplied fake/compatible engine when
    constructed with `engine`.
  - Returns normalized output:
    - `vitals`
    - `features`
    - `metadata`
- `ed_world_model/adapters/noop_emotion.py`
  - Added `NoopEmotionEngine`.
  - Returns a deep copy of the current patient emotion unchanged.
- `tests/test_hybrid_physiology_adapter.py`
  - Added focused tests for Hybrid adapter input construction, raw-text
    handling, diagnostic-result boundaries, no-action support, mutation
    isolation, existing Hybrid path compatibility with a fake LLM client, and
    normalized output.
- `tests/test_noop_emotion.py`
  - Added focused tests for unchanged emotion output, no vitals dependency,
    no state mutation, empty conversation input, and absence of behavior
    modifiers.
- `docs/milestones/M5_handoff.md`
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
- Orchestrator, ObservationBuilder, agents, and turn-loop code

No RuleBased adapter, PureLLM adapter, engine routing, agents, full turn loop,
nurse shadow execution, action validation changes, or observation changes were
implemented.

## Existing Hybrid Engine Interface Observed

- `transition_engines.hybrid_engine.HybridEngine`
  - Constructor:
    `HybridEngine(rule_engine: RuleBasedEngine, llm_client: LLMClient)`
  - Prediction method:
    `predict(engine_input: dict[str, Any], target_vital_names: Iterable[str] | None = None) -> EngineOutput`
- Current eval usage constructs it as:
  `HybridEngine(RuleBasedEngine(), llm_client)`.
- Existing Hybrid input tolerates missing `case_id`, `category`, and `pair_id`
  through `.get(...)` lookups, so the runtime adapter does not add those
  transition-pair/eval placeholders.
- Existing Hybrid prompt renders `action.raw_text`; passing `None` is accepted
  and renders as `null`.

## Adapter Behavior

`HybridPhysiologyAdapter.predict(global_state, engine_facing_action)`:

- Builds a fresh engine input from runtime state only:
  - `truth_state.scenario_description`
  - `truth_state.demographics`
  - `truth_state.patient_internal_state`
  - `patient_state.vitals`
  - `patient_state.features`
  - `patient_state.status_flags`
  - `known_facts`
  - `action.kind_hint`
  - `action.params`
  - `action.raw_text = None`
- Does not read `transitions/*.json`.
- Does not load transition pairs.
- Does not import or use pair labels, source fields, `modifier_text`, targets,
  or evaluation fields.
- Does not pass `truth_state.test_bank`.
- Allows released diagnostic results only through
  `known_facts.available_results`.
- Does not mutate `GlobalState` or the caller's action dictionary.
- Does not validate actions, call `StateManager`, build observations, decide
  `no_action`, create nurse shadow execution, or route between engines.

`NoopEmotionEngine.predict(...)`:

- Accepts current patient emotion plus optional conversation inputs.
- Returns unchanged patient emotion as a deep copy.
- Does not require or read vitals.
- Does not produce behavior modifiers.
- Does not mutate `GlobalState`.

## Tests Run and Results

Focused M5 run:

```bash
python -m pytest tests/test_hybrid_physiology_adapter.py tests/test_noop_emotion.py
```

Result: passed.

```text
14 passed in 0.11s
```

Adjacent focused run:

```bash
python -m pytest tests/test_state_manager.py::test_apply_physiology_update_updates_patient_state_only tests/test_state_manager.py::test_apply_physiology_update_records_event_when_requested tests/test_state_manager.py::test_apply_physiology_update_record_event_false_does_not_append_event tests/test_state_manager.py::test_apply_physiology_update_rejects_neither_and_both_inputs tests/test_action_registry.py tests/test_action_validator.py tests/test_orchestrator.py tests/test_observation_builder.py tests/test_hybrid_physiology_adapter.py tests/test_noop_emotion.py
```

Result: passed.

```text
119 passed in 0.12s
```

Broader adjacent run attempted:

```bash
python -m pytest tests/test_global_state.py tests/test_state_manager.py tests/test_action_registry.py tests/test_action_validator.py tests/test_orchestrator.py tests/test_observation_builder.py tests/test_hybrid_physiology_adapter.py tests/test_noop_emotion.py
```

Result: failed in three pre-existing `PatientEmotion.notes` cases unrelated to
M5 adapter behavior. The new M5 tests passed in that run.

```text
3 failed, 161 passed in 0.16s
```

Failing tests:

- `tests/test_global_state.py::test_patient_emotion_remains_separate_from_patient_profile`
- `tests/test_state_manager.py::test_update_patient_emotion_updates_patient_emotion_only`
- `tests/test_state_manager.py::test_partial_update_patient_emotion_preserves_prior_fields`

All three failures reject extra `notes` on `PatientEmotion`.

## Compatibility Issues With `raw_text=None`

No blocking compatibility issue was found.

- Existing `HybridEngine.predict(...)` accepts the adapter-built action with
  `raw_text: None`.
- Existing prompt construction renders it as `raw_text: null`.
- The adapter never fabricates runtime raw text, including for `no_action`.

## Deviations From `docs/v1_3_1.md`

- No intentional behavioral deviation.
- The adapter follows the M5 clarification that the physiology backend may use
  hidden patient-side truth from `truth_state.patient_internal_state`.
- The adapter intentionally excludes `truth_state.test_bank`; released
  diagnostic results are available only through `known_facts.available_results`.
- Runtime output is a simple dictionary instead of a dedicated physiology
  update model because no dedicated update model exists yet.

## Known Limitations

- `HybridPhysiologyAdapter` requires either an injected engine or an
  `llm_client`; it does not instantiate the real OpenAI client itself.
- There is no engine routing.
- There is no RuleBased-only adapter or PureLLM adapter.
- `target_vital_names` is not used in runtime adapter calls because runtime
  `GlobalState` has no transition-pair labels/evaluation targets.
- The adapter does not validate malformed clinician actions; that remains
  `ActionValidator` responsibility.
- The adapter does not apply output to state; later code should call
  `StateManager.apply_physiology_update(...)`.

## Exact Next Step

M6a Turn loop with stub agents.
