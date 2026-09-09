# M14o Emotion Augmentation Dataset Handoff

## Summary

Added `tools/build_emotion_augmentation_dataset.py`, an independent evaluation
data pipeline for the first three steps of the proposed emotion-engine benchmark:

1. Run randomly selected full simulations from `transitions/`.
2. Randomly extract patient/relative-directed source utterance candidates.
3. Use an LLM to rewrite candidates toward quota-controlled emotion/level cells.

This does not change runtime agents, `verbal_decision.py`, `emotion_engine.py`,
or existing scenario files.

## Outputs

Default output directory:

```text
eval_data/emotion_aug/
  runs/
  source_candidates.jsonl
  augmented_drafts.jsonl
  coverage_report.json
```

`augmented_drafts.jsonl` is silver draft data. Human reviewers still need to fill
or edit `human_label_next_state` before model evaluation.

## Candidate Schema

The script intentionally uses the simplified schema discussed with the user:

- `listener` is the recipient whose emotion will be evaluated.
- No `dialogue_so_far` field.
- `current_state` is taken from the previous turn's listener emotion.
- Only direct `recipient == patient` or `recipient == relative` messages are
  extracted.

## Verification

Ran:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python tools/build_emotion_augmentation_dataset.py \
  --run run_copde_pneumothorax_sonnet46_thinking_seed6.json \
  --skip-augmentation \
  --output-dir /private/tmp/emotion_aug_smoke \
  --candidates-per-run 5
```

The smoke test wrote 5 source candidates and a coverage report without calling
the LLM augmentation step.
