# M14r Emotion Augmentation Prompt Handoff

## Summary

Updated `tools/build_emotion_augmentation_dataset.py` so the augmentation prompt
is more explicit about primary target emotion generation.

## Change

The prompt now:

- Treats `target_emotion` / `target_level` as the primary target of the rewrite.
- Keeps one primary target per generated item for coverage accounting.
- Allows secondary emotions to change naturally when the same utterance would
  affect them.
- Adds the four emotion definitions from the emotion-engine prompt:
  `fear`, `sadness`, `overwhelm`, and `shame`.
- Adds level semantics for `not at all`, `low`, `high`, and `extreme`.
- Clarifies that `extreme` is reserved for catastrophic emotional crisis states.

The output schema is unchanged. `draft_label_next_state` remains a full
four-emotion state.

## Verification

Ran:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python -c "import json; from pathlib import Path; import tools.build_emotion_augmentation_dataset as b; c=json.loads(Path('eval_data/emotion_aug_pilot/source_candidates.jsonl').read_text(encoding='utf-8').splitlines()[0]); p=b._augmentation_prompt(c, target_emotion='fear', target_level='high'); assert 'Primary target:' in p; assert 'Emotion definitions:' in p; assert 'Level semantics:' in p; assert 'primary_target_emotion: fear' in p; print('\\n'.join(p.splitlines()[0:70]))"
```

No new simulations or LLM augmentation calls were run.
