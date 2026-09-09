# M14d Patient Stage 1 Speaking Handoff

## Scope

This patch loosens the patient Stage 1 speaking decision prompt so the patient
is more likely to speak when directly engaged.

No Stage 2 patient verbal prompt, orchestrator logic, turn loop behavior,
scenario JSON, transition engine, or existing test file was modified.

## Changes

- `ed_world_model/agents/patient.py`
  - Added direct-engagement guidance to `build_patient_decision_prompt(...)`.
  - If the most recent visible message is addressed to the patient and asks a
    question, and the patient is alive, conscious, and can speak, Stage 1 should
    default to `should_speak=true`.
  - If the patient is directly addressed with a treatment explanation,
    instruction, reassurance, or bedside action while still symptomatic or
    unstable, Stage 1 may allow a brief acknowledgment, symptom update, or
    concern.
  - If clinician/nurse messages are not addressed to the patient and do not
    require a patient answer, Stage 1 should usually stay silent.
  - If the patient is not alive, not conscious, or cannot speak, Stage 1 should
    set `should_speak=false`.

## Behavior Notes

- The two-stage patient design is preserved.
- Stage 2 remains the factual guard for final wording.
- The patient still may not invent symptoms, history, allergies, medications,
  social history, or review-of-systems findings.
- This is a prompt-level behavior change only; no hard-coded override or
  reprompt loop was added.

## Verification

```bash
python -m py_compile ed_world_model/agents/patient.py
```

Result: passed.

```bash
python -m pytest tests/test_patient_agent.py
```

Result: passed.

```text
45 passed in 0.13s
```

```bash
python -m pytest tests/test_agent_prompting.py
```

Result: passed.

```text
12 passed in 0.08s
```
