# M14w Clinician-to-Nurse Order Concision Handoff

## Summary

Added a clinician-to-nurse order communication policy so team-directed orders do
not become stacked vital/lab dumps.

## Change

Updated `ed_world_model/agents/clinician.py`.

Added `CLINICIAN_TO_NURSE_ORDER_POLICY`:

```text
Clinician-to-nurse order communication: keep team-directed orders natural and
concise. Use one order plus one short reason if needed. Do not stack multiple
vitals, labs, or pathophysiology details unless they are essential to the
immediate order. Use team communication, not a lab-value dump.
```

The policy is included in:

- clinician Stage 1 decision prompt
- clinician Stage 2 verbal prompt
- legacy monolithic clinician prompt path

This does not change patient/relative-facing verbal style diversity and does
not change the emotion engine.

## Verification

Ran:

```bash
python -m py_compile ed_world_model/agents/clinician.py
python -c "..."  # rendered Stage 1, Stage 2, and legacy clinician prompts
```

The rendered prompt check confirmed all three clinician prompt paths contain:

```text
Use team communication, not a lab-value dump.
```

The broader clinician/nurse/prompting tests were not rerun in this session
because they are already known to fail on old assertions expecting provider
`Profile traits` after M14v removed clinician/nurse trait-driven verbal style.
