# M14ae Emotion Prompt Strategy Experiments Handoff

## Summary

Added prompt-strategy variants for the emotion transition engine so zero-shot,
few-shot, and a single-call ReAct-style review can be compared without changing
the default deployed behavior.

## Files Modified

- `emotion_engine.py`
  - Added `PromptStrategy` / `PROMPT_STRATEGIES`.
  - Added `DEFAULT_PROMPT_STRATEGY`, read from
    `EMOTION_ENGINE_PROMPT_STRATEGY` with `zero_shot` as the default.
  - Added `build_step_prompt(...)`.
  - Added `few_shot` guidance with calibration examples.
  - Added `react` guidance as an internal single-call review/checklist.
  - Added optional `prompt_strategy` to `step_emotion(...)`.

- `ed_world_model/adapters/emotion_engine_adapter.py`
  - Added optional `prompt_strategy`.
  - Passed the strategy through to `step_emotion(...)`.

- `tools/evaluate_emotion_engine.py`
  - Added `--prompt-strategy {zero_shot,few_shot,react}`.
  - Kept baseline output names unchanged for `zero_shot`.
  - Adds the strategy name to output filenames for non-default variants.
  - Corrected stale no-scenario/no-traits help text.

- `generate_utterances.py`
  - Added `--emotion-prompt-strategy {zero_shot,few_shot,react}` for full
    simulation runs with the real emotion engine.
  - Writes `emotion_prompt_strategy` into saved run output when `--emotion` is
    used with `--output`.

## Prompt Strategy Semantics

- `zero_shot`: existing deployed prompt strategy; definitions + current state +
  clinical/dialogue context + transition rules + JSON contract.
- `few_shot`: zero-shot prompt plus synthetic calibration examples.
- `react`: zero-shot prompt plus a single-call internal Observe/Reason/Act/Check
  procedure. It does not use tools or emit a ReAct trace; the engine still
  returns only final JSON.

## Checks Run

```bash
python -m py_compile emotion_engine.py ed_world_model/adapters/emotion_engine_adapter.py tools/evaluate_emotion_engine.py generate_utterances.py
python -m pytest tests/test_emotion_rules_consistency.py
python tools/evaluate_emotion_engine.py --help
python generate_utterances.py --help
```

Also ran a local prompt-construction smoke check confirming:

- `zero_shot` prompt contains no calibration examples or ReAct guidance.
- `few_shot` prompt contains calibration examples.
- `react` prompt contains the ReAct-style guidance block.

## Not Run

No live LLM evaluation was run. It requires API credentials/cost, and the
`eval_data/emotion_aug0821` human labels have not yet been merged into
`human_labels_merged.json`.

## Suggested Next Commands

```bash
python tools/evaluate_emotion_engine.py \
  --provider anthropic \
  --model claude-opus-4-8 \
  --input-dir eval_data/emotion_aug0821 \
  --prompt-strategy zero_shot

python tools/evaluate_emotion_engine.py \
  --provider anthropic \
  --model claude-opus-4-8 \
  --input-dir eval_data/emotion_aug0821 \
  --prompt-strategy few_shot

python tools/evaluate_emotion_engine.py \
  --provider anthropic \
  --model claude-opus-4-8 \
  --input-dir eval_data/emotion_aug0821 \
  --prompt-strategy react
```

Before making human-gold claims for the Aug 21 set, fill and merge the human
labels into:

```text
eval_data/emotion_aug0821/human_labels_merged.json
```
