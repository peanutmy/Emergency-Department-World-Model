# M14e Clinician History Questions Handoff

## Scope

This patch applies the clinician-side part of increasing patient/relative
conversation: the clinician is prompted to ask more focused history questions
when clinically appropriate.

No orchestrator logic, turn-loop logic, patient Stage 1/Stage 2 logic, scenario
JSON, transition engine, or existing test file was modified.

## Changes

- `ed_world_model/agents/clinician.py`
  - Strengthened clinician Stage 1 decision guidance to avoid going idle while
    the patient remains unstable or unresolved.
  - Added a patient-question priority:
    - when the patient is alive, conscious, and can speak,
    - and no unresolved pending patient question exists,
    - strongly prefer one focused patient question if key facts are missing.
  - Focused question priorities are symptoms/onset/progression first, then
    allergies, past medical/surgical history, and home medications.
  - Added relative fallback:
    - when the patient is not conscious or cannot speak,
    - ask the relative one focused question about allergies, PMH/surgical
      history, home medications, baseline status, or recent course.
  - Extended pending-question guidance so the clinician avoids stacking new
    questions on patient or relative when one is already unresolved.
  - Updated clinician Stage 2 verbal prompt so patient/relative question
    decisions are rendered as exactly one short question with
    `requires_response=true`.

## Behavior Notes

- Urgent stabilization remains prioritized; the prompt says not to delay
  life-saving ABC care for a long history checklist.
- The single-action rule is preserved. A focused history question is still a
  verbal slot, not a second clinical action.
- The relative should become active through the existing pending-question path
  when the clinician asks a relative-directed question with
  `requires_response=true`.
- This is a prompt-level change only; no hard-coded fallback or reprompt was
  added.

## Verification

```bash
python -m py_compile ed_world_model/agents/clinician.py
```

Result: passed.

```bash
python -m pytest tests/test_clinician_agent.py
```

Result: passed.

```text
64 passed in 0.17s
```

```bash
python -m pytest tests/test_agent_prompting.py
```

Result: passed.

```text
12 passed in 0.10s
```
