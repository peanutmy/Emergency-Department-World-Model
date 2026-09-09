# M14z Extreme Terminal Prompt Handoff

## What Changed

- Updated `emotion_engine.py` step rules so `extreme` is explicitly a terminal,
  absorbing crisis state.
- Added runtime protection in `step_emotion`: if the current state for any
  emotion is `extreme`, the returned next state for that emotion is clamped back
  to `extreme` even if the LLM tries to lower it.
- Kept the stricter simulation rule that `extreme` ends the encounter, while
  allowing the prompt to recognize true listener breakdown from imminent death,
  catastrophic loss, total loss of control, or devastating humiliation/blame.
- Clarified de-escalation: non-extreme emotions may drop to `absent` when the
  specific trigger is explicitly resolved, even if the overall medical situation
  remains serious.
- Added concise application cues for sadness, shame, and overwhelm so the model
  evaluates all four dimensions instead of over-focusing on fear.
- Updated `tools/build_emotion_augmentation_dataset.py` extreme-target guidance
  to match the engine's terminal-state semantics.

## Checks

- `python -m py_compile emotion_engine.py tools/build_emotion_augmentation_dataset.py`
- `python -m pytest tests/test_emotion_rules_consistency.py`
- Smoke check: when an LLM stub tried to lower `fear` from `extreme` to
  `present`, `step_emotion` returned `fear: extreme` and appended a terminal-rule
  note to the reasoning.

## Notes

- Existing `eval_data/` outputs were not regenerated.
- The worktree had many pre-existing unrelated changes; this handoff only
  describes the emotion prompt/runtime changes above.
