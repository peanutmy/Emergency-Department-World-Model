# M14x Emotion Definition Prompt Update Handoff

## Summary

Updated the emotion-engine prompt semantics for fear, sadness, overwhelm, shame,
and the `extreme` level threshold.

## Changes

Changed `emotion_engine.py`:

- Added `EMOTION_DEFINITIONS_TEXT` as the shared four-emotion definition block.
- Updated the init prompt to use the shared definitions.
- Updated the step prompt to include the shared definitions.
- Updated sadness transition guidance to include imminent or deeply threatened
  loss, not only loss already incurred.
- Updated the `extreme` guidance so `extreme` is crisis-level and not simply
  "very high"; serious bad news usually reaches `high` unless the utterance
  explicitly supports panic, collapse, shutdown, catastrophic grief, total loss
  of control, or severe humiliation/blame.

Changed `tools/build_emotion_augmentation_dataset.py`:

- Reused `EMOTION_DEFINITIONS_TEXT` from `emotion_engine.py` in the augmentation
  prompt so generated augmented utterances use the same definitions as the
  engine.
- Updated the augmentation prompt's `extreme` level guidance and target
  instruction.

Changed `build_utterance_db.py`:

- Updated the concise emotion definitions and `extreme` phrase used by the
  older utterance DB builder so it no longer carries the old emotion boundaries.

## Verification

Ran:

```bash
python -m py_compile emotion_engine.py tools/build_emotion_augmentation_dataset.py build_utterance_db.py
python -c "import ed_world_model; import emotion_engine; from tools import build_emotion_augmentation_dataset; import build_utterance_db; print('ok')"
git diff --check -- emotion_engine.py tools/build_emotion_augmentation_dataset.py build_utterance_db.py
```

Rendered smoke checks confirmed:

- init prompt contains the new shared definitions
- step prompt contains the new shared definitions
- step rules contain the new sadness and `extreme` guidance
- augmentation prompt contains the shared definitions and updated `extreme`
  guidance

## Notes

The working tree already had many unrelated uncommitted/untracked files before
this change. They were not reverted or cleaned up.
