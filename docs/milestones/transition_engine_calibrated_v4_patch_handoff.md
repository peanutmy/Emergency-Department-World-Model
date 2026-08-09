# Transition Engine Calibrated V4 Patch Handoff

## Scope

This patch adds a new `calibrated_v4` Stage 1 prompt profile for the
direction-first hybrid transition engine. It preserves the existing
`calibrated_v3` profile and Stage 2 behavior.

## Files Modified

- `transition_engines/direction_first_hybrid_engine.py`
  - Added `calibrated_v4` to `STAGE1_PROMPT_PROFILES`.
  - Added profile dispatch for `calibrated_v4`.
  - Added `build_calibrated_v4_direction_prompt()`.
  - Added the V4 builder to `__all__`.

## Files Created

- `docs/milestones/transition_engine_calibrated_v4_patch_handoff.md`

## Files Intentionally Not Touched

- The `calibrated_v3` prompt implementation.
- The shared Stage 2 magnitude prompt.
- `transition_engines/run_direction_first_hybrid_eval.py`.
- Static few-shot examples and example selection.
- `transitions/` and `transitions_reviewed/`.
- Existing tests and evaluation outputs.

## Behavior

V4 retains V3's balanced direction policy and adds:

- A concise explanation of the two-stage workflow.
- An explicit warning that Stage 2 cannot correct a Stage 1 direction.
- A definition of the dead zone as an evaluation threshold around the
  pre-transition value rather than a normal clinical range.
- Clarification that values exactly on the dead-zone boundary are stable.
- Removal of the word `educational` from the V4 Stage 1 role description.

## Checks Run

```bash
python -m py_compile \
  transition_engines/direction_first_hybrid_engine.py \
  transition_engines/run_direction_first_hybrid_eval.py
```

Result: passed.

Prompt assertions verified:

- V3 source hash remained unchanged.
- V4 contains the workflow and dead-zone explanation.
- V4 omits `educational`.
- V4 receives sanitized case context.
- V4 does not receive the rule-based prediction.
- Stage 2 still receives the fixed direction and rule prediction.

```bash
python -m pytest \
  tests/test_few_shot.py \
  tests/test_few_shot_prompt.py \
  tests/test_clients_factory.py \
  tests/test_openai_llm_client.py \
  tests/test_llm_client.py \
  tests/test_run_model_comparison.py \
  tests/test_transition_engine_hybrid.py \
  tests/test_hybrid_eval_runner.py
```

Result: `86 passed`.

A three-pair fake-client smoke evaluation and a complete 57-pair fake-client
evaluation both completed with zero Stage 1 and Stage 2 errors.

## Deviations From `docs/v1_3_1.md`

This is an explicitly requested transition-engine evaluation experiment. It
does not change the `ed_world_model` runtime architecture or milestone
implementation.

## Known Limitations

- No real API evaluation has been run for V4.
- Fake-client accuracy is not meaningful; fake mode verifies wiring only.
- V4 changes Stage 1 framing only. Stage 2 and static few-shot examples are
  unchanged.
- The prompt cannot resolve information absent from standalone transition
  inputs.

## Exact Next Step

Run the real GPT-5, three-shot static V4 evaluation on
`transitions_reviewed/` into a new output directory, then compare its
vital-level confusion matrix, normalized RMSE, and failure distribution with
V3.
