# M14aj Emotion Missing Fields Handoff

## Scope

The user explicitly requested retaining each emotion's pre-update level when
the model omits that field. Read AGENTS.md and the preceding M14ai handoff.
This request authorizes the scoped change to the existing emotion engine.

## Implementation

- `_coerce_step_output` now requires the current `EmotionState` and starts with
  a copy of its levels before applying model-provided non-null emotion values.
- `step_emotion` supplies the state immediately before the utterance's update.
- Missing and null fields retain their prior levels. An empty JSON object or
  empty `new_state` preserves all levels. Explicit valid levels are applied,
  subject to the existing absorbing-extreme rule. Invalid levels still fail
  validation.
- The adapter uses this engine path for both patient and relative updates, so
  consecutive utterances retain levels from the most recent successful update.
- No prompt text, initial-state defaults, adapter code, or existing tests were
  changed in this session. Earlier pending edits and documents were preserved.

## Validation

- 2,608 offline in-memory regression cases passed: all 81 initial states and
  all 16 field subsets with omitted/null fields, explicit updates, extreme
  clamping, original trajectory preservation, empty results, flat responses,
  invalid levels, and sequential adapter updates for both roles and all three
  prompt strategies. No live model calls were made.
- `python -B -m pytest -p no:cacheprovider tests/test_emotion_rules_consistency.py tests/test_noop_emotion.py -q -k 'not test_step_prompt_embeds_shared_definitions_verbatim'`:
  16 passed, 1 deselected. The excluded assertion was already obsolete after
  the earlier prompt-variable refactor; it checks the raw template instead of
  the rendered prompt. Existing tests remain untouched under repository rules.
- `git diff --check` passed. The full repository suite was not run.

## Documentation Follow-up

Earlier manuscript and audit documents describe missing fields defaulting to
`absent`. That description is historical after this fix. Paper text should now
say that omitted or null emotion fields retain their pre-update levels, while
explicit values undergo validation and the existing extreme-state constraint.
The existing Markdown/Word research drafts were not revised in this scoped
runtime change. Align them with the implementation used for the experiments
before submission.
