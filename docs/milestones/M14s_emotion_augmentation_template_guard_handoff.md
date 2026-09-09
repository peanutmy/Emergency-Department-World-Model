# M14s Emotion Augmentation Template Guard Handoff

## Summary

Tightened `tools/build_emotion_augmentation_dataset.py` after generated
augmentation drafts repeatedly used generic emotional preambles such as
`I need to be straight with you` and `I need you to listen to me very carefully`.

## Change

The augmentation prompt now explicitly says not to increase emotion by simply
adding a generic urgency preamble. It instructs the model to make the target
emotion arise from specific clinical content, disclosure, question, reassurance,
loss, complexity, or dignity/judgment cues.

Added deterministic draft validation that rejects augmented utterances starting
with common generic emotional openings, after allowing an optional salutation
such as `Sir,`.

Existing generated data under `eval_data/` was not regenerated in this patch.
The new prompt and guard affect future augmentation runs.

## Verification

Ran:

```bash
python -m py_compile tools/build_emotion_augmentation_dataset.py
python -c "from tools.build_emotion_augmentation_dataset import _has_generic_emotional_opening as h; ..."
python -c "from tools.build_emotion_augmentation_dataset import _valid_draft; ..."
python -c "import json; from pathlib import Path; import tools.build_emotion_augmentation_dataset as b; ..."
```

The checks confirmed generic openings are rejected, normal clinical utterances
still pass, and the rendered prompt includes the new anti-template instruction.
