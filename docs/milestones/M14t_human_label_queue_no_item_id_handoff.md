# M14t Human Label Queue No Item ID Handoff

## Summary

Removed `item_id` from the human-labeling queue view.

## Change

Updated `tools/build_emotion_augmentation_dataset.py` so
`human_label_queue.jsonl` rows now contain only:

- `listener`
- `speaker`
- `traits`
- `current_state`
- `augmented_utterance`
- `human_label_next_state`

`augmented_drafts.jsonl` still retains `item_id` and source provenance fields
for debugging and internal traceability.

Updated current generated queue files:

- `eval_data/emotion_aug_48/human_label_queue.jsonl`
- `eval_data/emotion_aug_48/human_label_queue_readable.json`
- `eval_data/emotion_aug_pilot/human_label_queue.jsonl`

## Verification

Ran:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python -c "..."  # verified 48 JSONL rows and no item_id
python -c "..."  # verified 48 readable JSON rows and no item_id
rg -n '"item_id"|item_id' eval_data/emotion_aug_48/human_label_queue.jsonl eval_data/emotion_aug_48/human_label_queue_readable.json eval_data/emotion_aug_pilot/human_label_queue.jsonl docs/milestones/M14q_human_label_queue_handoff.md
```

The `rg` command exited with no matches.
