# M11 Patient Truth Deterioration Patch Handoff

## Patch Summary

- Patient prompt now frames the patient as a source of patient-side truth.
- Patient prompt now forbids inventing symptoms, history, allergies,
  medications, or ROS findings beyond the observation.
- Patient prompt now forbids adding plausible disease-associated symptoms just
  because they fit the diagnosis.
- Patient prompt now directs conservative answers for unlisted symptoms or
  facts.
- Clinician prompt now includes a general deterioration/worsening response rule
  prioritizing reassessment and stabilizing/escalating actions over non-urgent
  history questions.
- No clinical guidance library, family-level indications, kind_hint-level
  indications, validator medical-correctness logic, behavior actions, task
  queues, resource availability, or nurse physical actions were added.

## Files Modified

- `ed_world_model/agents/patient.py`
  - Added prompt-only patient truthfulness rules.
  - Clarified that `hidden_*` fields are patient-owned truth but must not be
    expanded beyond the observation.
- `ed_world_model/agents/clinician.py`
  - Added a prompt-only deterioration response rule.
- `tests/test_patient_agent.py`
  - Added focused prompt assertions for no invented patient facts, no plausible
    disease-associated symptoms, and conservative answers for unlisted
    symptoms/facts.
- `tests/test_clinician_agent.py`
  - Added focused prompt assertion for deterioration response guidance.
- `docs/milestones/M11_patient_truth_deterioration_patch_handoff.md`
  - Added this handoff.

No files under `transition_engines/` or `transitions/` were modified by this
patch. The worktree already showed a modified transition JSON before this work;
it was left untouched.

## Tests Run and Results

```bash
python -m pytest tests/test_patient_agent.py tests/test_clinician_agent.py
```

Result: passed.

```text
75 passed in 0.14s
```
