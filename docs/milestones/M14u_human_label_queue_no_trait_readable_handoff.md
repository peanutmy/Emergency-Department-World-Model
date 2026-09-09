# M14u Human Label Queue No Trait Readable Handoff

## Summary

Created a no-trait readable human-label queue file for the 48-item emotion
augmentation set.

## Output

Generated:

```text
eval_data/emotion_aug_48/human_label_queue_readable_no_trait.json
```

This file is derived from:

```text
eval_data/emotion_aug_48/human_label_queue_readable.json
```

Each row omits `traits` and contains only:

- `listener`
- `speaker`
- `current_state`
- `augmented_utterance`
- `human_label_next_state`

The original readable file with traits was left unchanged.

## Verification

Ran a JSON validation check confirming:

- 48 rows
- no row contains `traits`
- first-row keys are `augmented_utterance`, `current_state`,
  `human_label_next_state`, `listener`, and `speaker`
