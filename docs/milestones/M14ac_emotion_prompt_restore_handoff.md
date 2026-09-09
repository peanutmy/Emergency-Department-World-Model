# M14ac Emotion Prompt Restore Handoff

## Summary

Restored two `emotion_engine.py` prompt edits per user instruction.

## Files Modified

- `emotion_engine.py`
  - Restored the de-escalation sentence so a concrete reassurance, concrete plan
    or next step, symptom relief, or resolved uncertainty can address the
    underlying trigger.
  - Restored the clinical context header to describe how the listener is doing
    clinically and that this constrains how they can react.

## Files Intentionally Not Touched

- `tools/build_emotion_augmentation_dataset.py`
- `transition_engines/`
- `transitions/`
- generated `eval_data/` outputs

## Checks Run

```bash
python -m py_compile emotion_engine.py tools/build_emotion_augmentation_dataset.py
python -m pytest tests/test_emotion_rules_consistency.py
```

## Deviations From `docs/v1_3_1.md`

None.

## Known Limitations

The broader working tree was already dirty. This handoff only describes the two
prompt restorations above.

## Exact Next Step

Use the existing all-scenarios augmentation command from
`M14ab_emotion_aug_all_scenarios_handoff.md` when ready to generate the 120-item
human-label queue.
