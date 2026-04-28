# M3 Observation Gateway Context

## Goal

Build role-specific observations that enforce information boundaries across clinician, nurse, patient, and relative.

## Relevant Sections From V4

- Section 5: Observation Gateway role views.
- Section 20.2: subjective state feeds patient observation.
- Section 20.8: ECG/imaging should fit future diagnostic result representation.
- Section 18: pitfalls about vitals, hidden state, and conversation memory.

## Allowed Files/Folders

- `ed_multiagent/observation/`
- `ed_multiagent/emsim_adapter/subjective.py`
- `ed_multiagent/memory/`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not expose hidden state, hidden diagnosis, raw pathology, undiscovered allergy, or drug PD internals to clinician/nurse.
- Do not expose numeric vitals to patient unless staff communicated them.
- Do not mutate memory or world state while building observations.
- Do not add LLM calls.

## Required Tests

- Clinician view includes discovered facts and omits hidden truth.
- Nurse view includes current vitals and task queue.
- Patient view includes symptoms/subjective state and omits numeric vitals by default.
- Relative view includes private family knowledge and visible patient state only.

## Acceptance Criteria

- Observation builders are deterministic and side-effect free.
- Each role gets only the data permitted by the v4 design.
