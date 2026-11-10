# M14af Emotion Prompt Variable Handoff

## Scope

The user clarified that only the emotion definitions in the prompt should be
editable variables, and authorized this implementation. Dynamic emotion state
fields and additional tracked emotions are outside this change.

Read AGENTS.md and the M14ae and M14ad handoffs before editing, as well as the
M14x handoff documenting the existing shared definition text.

## Changes

- `emotion_engine.py`: replaced import-time concatenation of
  `EMOTION_DEFINITIONS_TEXT` into `STEP_PROMPT_TEMPLATE` with an explicit
  `{emotion_definitions}` placeholder. `build_step_prompt` supplies the current
  text when formatting the prompt.
- Added a comment identifying `EMOTION_DEFINITIONS_TEXT` as the editing point.
  Its original wording, state schemas, transition rules, and output fields are
  unchanged. The augmentation builder still imports this same definition block.
- To customize definitions, edit `EMOTION_DEFINITIONS_TEXT` and restart the
  process. Adding another tracked emotion still requires schema changes.

## Validation

- Python syntax compilation passed.
- A dependency-free check executed the actual prompt constants and builder
  functions extracted using Python AST. Rendered zero-shot, few-shot, and ReAct
  prompts matched the pre-change Git HEAD versions byte for byte.
- The same check replaced the definitions with custom text containing literal
  braces and confirmed correct insertion in all three prompt strategies.
- `git diff --check` passed.
- Initial full engine import and pytest execution were blocked by missing
  dependencies. The user subsequently authorized installing `pydantic` and
  `pytest`; installation succeeded with versions 2.13.5 and 9.1.1 respectively.
- Imports of `ed_world_model` and `emotion_engine` now pass.
- `python -m pytest tests/test_emotion_rules_consistency.py -q`: 11 passed,
  1 failed. The failure is the outdated raw-template assertion described below.
- An actual `step_emotion` smoke check using a fake LLM passed for all three
  strategies, verifying that both default and edited definitions reach the
  LLM prompt and that literal braces in edited definitions are preserved.
  No live LLM calls were made.

## Existing Test Follow-up

Existing test files were left untouched per AGENTS.md. The assertion in
`tests/test_emotion_rules_consistency.py::test_step_prompt_embeds_shared_definitions_verbatim`
expects the definitions inside the unformatted template and is now outdated:
the raw template intentionally contains a placeholder. When test edits are
authorized, change this assertion to check the result of `build_step_prompt`.
The direct rendered-prompt checks above already verify that behavior.
