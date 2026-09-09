# M14y Three-Level Emotion Schema Handoff

## Summary

Changed the emotion level schema from four levels
`not at all / low / high / extreme` to three levels
`absent / present / extreme`.

## Changes

Changed `emotion_engine.py`:

- `EmotionLevel = Literal["absent", "present", "extreme"]`
- `EMOTION_LEVELS = ("absent", "present", "extreme")`
- `EmotionState` defaults now use `absent`.
- Init and step prompts now describe the ordinal scale as
  `absent < present < extreme`.
- Step rules now treat serious bad news as at least `present`, while preserving
  the stricter `extreme` threshold.

Changed `ed_world_model/state/global_state.py`:

- `AgentEmotion` now uses the same three-level schema as `emotion_engine`.
- Defaults now use `absent`.

Changed `ed_world_model/adapters/emotion_engine_adapter.py`:

- Updated comments/docstrings from `not at all` to `absent`.

Changed `tools/build_emotion_augmentation_dataset.py`:

- Reachability logic now uses `present -> extreme` for terminal escalation.
- `absent` replaces `not at all`.
- Prompt level semantics and JSON output schema now use
  `absent|present|extreme`.

Changed `build_utterance_db.py` and `examples/run_emotion_trajectory.py`:

- Updated legacy/example prompt and fake LLM level values to the three-level
  schema.

## Verification

Ran:

```bash
python -m py_compile emotion_engine.py ed_world_model/state/global_state.py ed_world_model/adapters/emotion_engine_adapter.py ed_world_model/adapters/noop_emotion.py tools/build_emotion_augmentation_dataset.py tools/evaluate_emotion_engine.py build_utterance_db.py examples/run_emotion_trajectory.py
python -c "import ed_world_model; import emotion_engine; from tools import build_emotion_augmentation_dataset, evaluate_emotion_engine; import build_utterance_db; import examples.run_emotion_trajectory; print('ok')"
python examples/run_emotion_trajectory.py --scenario "transitions/Cardiology/Nightmares Case 7 Hyperkalemia.json" --emotion fake
git diff --check -- emotion_engine.py ed_world_model/state/global_state.py ed_world_model/adapters/emotion_engine_adapter.py tools/build_emotion_augmentation_dataset.py build_utterance_db.py
```

Also ran a direct smoke check confirming:

- `EMOTION_LEVELS == ("absent", "present", "extreme")`
- `EmotionState()` defaults to all `absent`
- `AgentEmotion()` defaults to all `absent`
- adapter conversion accepts `present` and `extreme`
- old `not at all` is rejected by the new runtime schema

## Notes

Existing historical eval artifacts under `eval_data/` still contain old
four-level labels. They were not rewritten. To compare against those older
artifacts, map `not at all -> absent`, `low/high -> present`, and
`extreme -> extreme` first, or regenerate predictions/labels using the new
three-level schema.

The working tree already had many unrelated dirty/untracked files. They were
not reverted or cleaned up.
