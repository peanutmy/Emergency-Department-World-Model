# M14q Human Label Queue Handoff

## Summary

Updated `tools/build_emotion_augmentation_dataset.py` to write a concise
human-labeling file in addition to the full debug/provenance draft file.

## Output

The script now writes:

```text
<output-dir>/augmented_drafts.jsonl
<output-dir>/human_label_queue.jsonl
```

`augmented_drafts.jsonl` remains the full debug record with source utterance,
target cell, draft label, rationale, and tags.

`human_label_queue.jsonl` is the file intended to send to human labelers. Each
row contains only:

- `listener`
- `speaker`
- `traits`
- `current_state`
- `augmented_utterance`
- `human_label_next_state`

`human_label_next_state` is initialized as:

```json
{
  "fear": "",
  "sadness": "",
  "overwhelm": "",
  "shame": ""
}
```

## Verification

Ran:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python tools/build_emotion_augmentation_dataset.py \
  --skip-simulations \
  --skip-augmentation \
  --output-dir eval_data/emotion_aug_pilot
```

This generated `eval_data/emotion_aug_pilot/human_label_queue.jsonl` with 4
rows from the existing pilot drafts.

## Follow-up Patch

Removed `scenario_summary` from `human_label_queue.jsonl` rows to keep the
human-labeling file focused on the utterance, listener/speaker metadata, traits,
current emotion state, and blank human labels. The full `augmented_drafts.jsonl`
debug/provenance file still retains `scenario_summary`.

Follow-up verification:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python -c "import json, pathlib; rows=[json.loads(line) for line in pathlib.Path('eval_data/emotion_aug_pilot/human_label_queue.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]; assert rows; assert all('scenario_summary' not in row for row in rows); print(len(rows), sorted(rows[0].keys()))"
rg -n "scenario_summary" eval_data/emotion_aug_pilot/human_label_queue.jsonl
```

The JSONL validation found 4 rows and confirmed no row contains
`scenario_summary`. The `rg` command exited with no matches.
