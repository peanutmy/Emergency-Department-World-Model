# M14p Emotion Step Output Coercion Handoff

## Summary

Fixed a common emotion-engine LLM parsing failure:

```text
Extra inputs are not permitted [type=extra_forbidden, input_value='No emotion changes ...']
```

The failure happens when the LLM puts a `reasoning` sentence inside
`new_state`. `EmotionState` correctly forbids extra fields, but the adapter then
freezes the emotion for that utterance.

## Change

Updated `emotion_engine.py`:

- Added `_coerce_step_output(raw)`.
- It keeps strict validation for the four emotion levels.
- It tolerates misplaced `reasoning`, `rationale`, or `evidence` fields by
  moving/stringifying metadata before validating `StepOutput`.
- `step_emotion(...)` now calls `_coerce_step_output(raw)` instead of validating
  raw LLM JSON directly.

## Verification

Ran:

```bash
python -m py_compile emotion_engine.py tools/build_emotion_augmentation_dataset.py
python -c 'from emotion_engine import _coerce_step_output; raw={"new_state":{"fear":"low","sadness":"not at all","overwhelm":"low","shame":"not at all","reasoning":"No emotion changes this tick because treatment is already managing."},"evidence":"routine care"}; parsed=_coerce_step_output(raw); print(parsed.model_dump())'
```

The malformed shape now parses into a valid `StepOutput`.
