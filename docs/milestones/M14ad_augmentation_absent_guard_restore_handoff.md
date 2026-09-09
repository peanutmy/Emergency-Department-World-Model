# M14ad Augmentation Absent Guard Restore Handoff

## Summary

Restored the `present -> absent` augmentation prompt wording after user
clarified that augmented utterances may invent plausible resolution content.

## Files Modified

- `tools/build_emotion_augmentation_dataset.py`
  - Removed the added guard that forbade invented normal tests, successful
    treatment response, completed procedures, family facts, or fully-safe claims
    for `present -> absent`.
  - Restored the shorter prior wording: the end state must be genuine absence,
    with the trigger fully resolved or gone.

- `tests/test_emotion_rules_consistency.py`
  - Removed the test that required the deleted guard text.

## Files Intentionally Not Touched

- `emotion_engine.py`
- `transition_engines/`
- `transitions/`
- generated `eval_data/` outputs

## Checks Run

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py tests/test_emotion_rules_consistency.py
python -m pytest tests/test_emotion_rules_consistency.py
```

## Deviations From `docs/v1_3_1.md`

None.

## Known Limitations

The broader working tree was already dirty. This handoff only describes the
augmentation prompt guard restoration above.

## Exact Next Step

Continue using `tools/build_emotion_augmentation_dataset.py` for the
all-scenarios augmentation run when ready.
