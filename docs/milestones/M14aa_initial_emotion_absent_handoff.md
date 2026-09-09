# M14aa Initial Emotion Absent Handoff

## What Changed

- Removed the LLM-based initial emotion prompt from `emotion_engine.py`.
- Removed the obsolete `InitOutput` schema and init-only prompt formatting
  helpers.
- Changed `init_emotion()` to return a deterministic all-`absent` emotion state:
  `fear`, `sadness`, `overwhelm`, and `shame` all start as `absent`.
- Preserved the `init_emotion()` function signature for backwards
  compatibility, but it no longer calls `llm_call` or uses scenario, vitals, or
  traits to infer opening emotion.
- Updated adapter/example comments so they no longer describe initial emotion as
  an LLM-seeded state.

## Checks

- `python -m py_compile emotion_engine.py ed_world_model/adapters/emotion_engine_adapter.py examples/run_emotion_trajectory.py`
- `python -m pytest tests/test_emotion_rules_consistency.py`
- Smoke check confirmed `init_emotion()` returns all `absent` and makes zero
  LLM calls.

## Notes

- Existing prediction/metrics files were not regenerated.
- Emotion changes are now driven only by received utterances through
  `step_emotion()`.
