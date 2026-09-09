# M14j Emotion Clinical Escalation Handoff

## Scope

Adjusted the emotion engine step prompt so serious clinical disclosures to the
patient can actually escalate emotion instead of being repeatedly buffered to
`fear=low`.

No scenario JSON, transition engines, runtime state schema, or tests were
modified.

## Problem

In the intra-abdominal sepsis run, the emotion engine was being called
successfully, but it kept returning:

```text
fear=low, sadness=not at all, overwhelm=not at all, shame=not at all
```

Even when the clinician directly told the patient that liver and bile ducts were
under significant stress and that more urgent investigation was needed. The
debug output showed this was not an adapter `last_error` problem; it was prompt
policy. The previous step prompt over-weighted `DEFAULT TO NO CHANGE` and the
presence of a calm plan.

## Changes

- `emotion_engine.py`
  - Added a `CLINICAL THREAT ESCALATION` rule to `STEP_PROMPT_TEMPLATE`.
  - Direct disclosure of serious abnormality or deterioration is now explicitly
    a concrete reason to raise fear by one level unless already high/extreme.
  - Examples include organ stress, bile duct blockage, infection/sepsis/septic
    shock, dangerous BP/O2, lactate/acidosis, vasopressors, intubation, ICU,
    urgent procedures, and surgery.
  - Clear clinician planning can buffer the size of the rise, but should not
    cancel the rise.
  - Multi-finding or technical urgent disclosures can raise overwhelm one level.

## Verification

```bash
python -m py_compile emotion_engine.py generate_utterances.py
```

Result: passed.

Prompt text check confirmed `CLINICAL THREAT ESCALATION` and the `low->high
fear` instruction are present in `STEP_PROMPT_TEMPLATE`.

## Notes

This change has not been verified with a live LLM rerun in this session to avoid
hitting the active Anthropic token-per-minute limit again. The next intra-
abdominal sepsis run should be inspected at the first direct serious disclosure
to confirm fear moves from `low` to `high` and overwhelm can move from `not at
all` to `low` when the disclosure is technical or urgent.
