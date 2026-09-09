# M14k Emotion Threat Floor Handoff

## Scope

Added a deterministic clinical-threat floor to the emotion engine so direct
serious clinical disclosures cannot be neutralized back to no emotional change
by high-trust traits or clinician reassurance.

No scenario JSON, transition engines, runtime state schema, or tests were
modified.

## Problem

`run_sepsis_v2.json` showed the emotion engine was not failing:

- `patient_emotion_last_error` was `null` on every turn.
- No `emotion_engine_error` events were present.
- Turn 3 and Turn 4 `notes` showed the LLM understood the serious disclosure but
  kept fear at `low` because the patient had high provider trust and the
  clinician was reassuring.

This meant prompt-only guidance was insufficient. The LLM ignored the intended
rule that reassurance can buffer severity but must not cancel a one-level rise
after serious clinical disclosure.

## Changes

- `emotion_engine.py`
  - Added clinical-threat term detection for direct disclosures such as:
    `significant stress`, bile duct blockage, infection/sepsis/septic shock,
    dangerous/critically low vitals, lactate/acidosis, vasopressors,
    norepinephrine, intubation, ICU, urgent care, surgery, and escalation of
    care.
  - Added `_apply_clinical_threat_floor(...)` after LLM output validation in
    `step_emotion(...)`.
  - If a direct clinical threat is present and current fear is below `high`, the
    returned fear is floored at one level above the current fear.
  - If the disclosure is technical or combines diagnosis/findings/urgent care,
    overwhelm is floored one level above current overwhelm unless already
    high/extreme.
  - Appends a `Clinical threat floor: ...` sentence to `last_reasoning` whenever
    the deterministic floor corrects the LLM output.

## Verification

```bash
python -m py_compile emotion_engine.py generate_utterances.py
```

Result: passed.

Fake LLM regression check:

- Current state: `fear=low`, `overwhelm=not at all`
- Utterance: liver and bile ducts under significant stress, ultrasound ordered
- Fake LLM output: still `fear=low`, `overwhelm=not at all`

Result after `step_emotion(...)`:

```text
fear=high
overwhelm=low
```

Routine question check:

- Utterance: `Do you have any allergies to medications?`
- Fake LLM output: `fear=low`

Result: no floor applied; state remained unchanged.
