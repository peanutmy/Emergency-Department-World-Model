# M14ab Emotion Aug All-Scenarios Handoff

## Summary

Prepared the emotion augmentation pipeline for a real run across every scenario
and a 120-item human-label queue.

## Files Modified

- `emotion_engine.py`
  - Changed remaining patient-only wording to listener-aware wording where the
    listener may be a patient or relative.
  - Tightened de-escalation semantics so a plan or next step lowers emotion only
    when it directly resolves the specific trigger.
  - Clarified that clinical context helps interpret perception/understanding; it
    is not an independent emotion trigger.
  - Allowed no-change evidence wording in the step prompt.

- `tools/build_emotion_augmentation_dataset.py`
  - Added `--all-scenarios` to run each discovered scenario once in sorted order,
    instead of random scenario sampling.
  - Updated the example command to use `--all-scenarios` and
    `--target-min-per-cell 10` for 120 generated drafts.
  - Tightened `present -> absent` generation instructions so the LLM does not
    invent unsupported normal tests, successful treatment response, completed
    procedures, family facts, or full-safety claims.
  - Split `absent -> absent` wording from other no-change cases so it no longer
    refers to an existing trigger.

- `tests/test_emotion_rules_consistency.py`
  - Added tests for the absent prompt rules and the new all-scenarios run mode.

## Files Intentionally Not Touched

- `transition_engines/`
- `transitions/`
- scenario JSON content
- generated `eval_data/` outputs
- full runtime agent/orchestration code outside the prompt path

## Checks Run

```bash
python -m py_compile emotion_engine.py tools/build_emotion_augmentation_dataset.py tools/evaluate_emotion_engine.py generate_utterances.py ed_world_model/scenario_loader.py ed_world_model/adapters/hybrid_physiology_adapter.py ed_world_model/adapters/emotion_engine_adapter.py
python -m pytest tests/test_emotion_rules_consistency.py
python -c "import ed_world_model; import emotion_engine; from tools import build_emotion_augmentation_dataset, evaluate_emotion_engine; import generate_utterances; print('ok')"
python tools/build_emotion_augmentation_dataset.py --help
git diff --check -- emotion_engine.py tools/build_emotion_augmentation_dataset.py tests/test_emotion_rules_consistency.py
```

Also ran an offline loader check across all scenarios:

- `loaded=49`
- `missing_scenario_description=0`
- `missing_test_bank=2`

The two empty-test-bank scenarios are:

- `transitions/OB-GYN/Postpartum Hemorrhage and NRP.json`
- `transitions/Resuscitation/Anaphylaxis and Medication Error.json`

Both have empty `case_context.supporting_findings`; no scenario files were
changed.

## Deviations From `docs/v1_3_1.md`

None. Existing transition engines and scenario files were not modified.

## Known Limitations

- The real all-scenarios run was not started in this session because it requires
  networked LLM calls and substantial token/cost budget.
- `--target-min-per-cell 10` aims for 120 drafts across the 12 emotion/level
  cells. If some LLM augmentation attempts fail validation, the final count can
  be lower; check `coverage_report.json`.
- Two scenarios have empty diagnostic test banks, so they will not release
  diagnostic results during simulation unless their scenario data is later
  updated.

## Exact Next Step

Run the real generation job with a fresh output directory:

```bash
env AGENT_LLM_MODEL="claude-sonnet-4-6" \
ANTHROPIC_THINKING_TYPE=adaptive \
ANTHROPIC_THINKING_EFFORT=max \
ANTHROPIC_THINKING_DISPLAY=omitted \
python tools/build_emotion_augmentation_dataset.py \
  --provider anthropic \
  --all-scenarios \
  --turns 15 \
  --physiology-mode hybrid \
  --target-min-per-cell 10 \
  --augment-attempts 4 \
  --candidates-per-run 8 \
  --output-dir eval_data/emotion_aug_all_scenarios_120
```

After it completes, inspect:

- `eval_data/emotion_aug_all_scenarios_120/coverage_report.json`
- `eval_data/emotion_aug_all_scenarios_120/human_label_queue.jsonl`
- `eval_data/emotion_aug_all_scenarios_120/augmented_drafts.jsonl`
