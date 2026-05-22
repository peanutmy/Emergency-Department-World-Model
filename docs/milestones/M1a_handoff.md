# M1a Handoff

## Files Created/Modified

- `ed_world_model/state/global_state.py`
  - Added Pydantic v2 GlobalState model tree:
    - `GlobalState`
    - `TruthState`
    - `PatientInternalState`
    - `TestBankItem`
    - `PatientState`
    - `Vitals`
    - `Features`
    - `StatusFlags`
    - `KnownFacts`
    - `DiagnosticResult`
    - `PsychState`
    - `PatientEmotion`
    - `RuntimeState`
    - `Message`
    - `Event`
  - `PendingQuestion`
  - `PendingDiagnosticResult`
  - Added `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2` for later runtime use.
  - Added strict schema defaults for the GlobalState model tree.
  - Clarified in model documentation that `PatientInternalState` is stable hidden patient-side truth, that its `symptoms` are static dialogue facts, and that `disclosure_rules` is free-form prompt guidance rather than a deterministic rule engine.
  - Updated `PatientEmotion.intensity` to the categorical values `low`, `medium`, `high`, or `None`.
  - Clarified in model documentation that `Event.payload` is event-specific metadata for logging, observation, and debugging only, not durable state.
  - Removed the recursive `raw_text` key guard. `raw_text` belongs to legacy transition-pair extraction/evaluation data and later runtime action/adapter boundaries, not the M1a GlobalState model layer.
- `tests/test_global_state.py`
  - Added focused tests for M1a model construction, defaults, diagnostic test bank items, message/event separation, known facts, serialization, GlobalState construction without `raw_text`, static patient-internal truth, categorical patient emotion intensity, and event payload metadata boundaries.
- `docs/milestones/M1a_handoff.md`
  - Added this handoff.

Pytest generated local `__pycache__` entries during the focused run; these are not M1a source files.

## Tests Run

```bash
python -m pytest tests/test_global_state.py
```

Result: passed.

```text
10 passed in 0.06s
```

Broad pytest was not run because M1a only adds GlobalState models and focused tests were requested.

## Deviations From `docs/v1_3_1.md`

- No behavioral deviations were intentionally introduced.
- `docs/v1_3_1.md` names state sections but does not specify every field for `Vitals`, `Features`, `PatientEmotion`, `Message`, or `Event`.
  - `Vitals` uses the existing canonical transition-engine vital keys: `HR`, `BP_sys`, `BP_dia`, `RR`, `O2Sat`, and `T`.
  - `Features` is intentionally open to arbitrary feature keys.
  - `PatientEmotion`, `Message`, and `Event` are compact minimal schemas for later adapters/managers to use.
  - `PatientEmotion.intensity` is categorical: `low`, `medium`, `high`, or `None`.
- `RuntimeState.max_turns` defaults to `50` so `GlobalState()` is constructable and not immediately terminated by default. The design doc requires `max_turns` but does not specify a default.
- `GlobalState` does not define runtime action objects, so it does not enforce `raw_text = null`. That check belongs later in `ActionValidator` and/or `PhysiologyAdapter` tests.

## Known Limitations

- M1a contains models only.
- `StateManager` is not implemented.
- Diagnostic ordering, result release, and fallback use of `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS` are not implemented yet.
- `PatientInternalState` is only modeled as stable hidden patient-side truth. No physiology or emotion runtime behavior updates it.
- `Event.payload` remains generic metadata only and is not recursively scanned for `raw_text`; no StateManager behavior exists yet to enforce event lifecycle or durability boundaries.
- Action registry, action validation, orchestration, observation building, adapters, agents, trajectory logging, and the turn loop are not implemented.
- There is no memory implementation; `known_facts` remains the only clinical-team factual store.
- No existing transition engine or scenario files were modified.

## Exact Next Step

M1b StateManager.
