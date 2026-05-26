# M11 Prompt Patch Handoff

## Files Modified

- `ed_world_model/agents/clinician.py`
  - Added clinician prompt policy to balance urgent stabilization with focused
    history-taking.
  - Prompt now allows immediate airway/breathing/circulation stabilization
    before a long history checklist.
  - Prompt now directs the clinician to ask one brief focused question when
    clinically appropriate if the patient is conscious and `can_speak=True`.
  - Prompt now explicitly says a focused history question is not an additional
    clinical action and may be paired with an urgent treatment or diagnostic
    structured action as the single `verbal_action`.
  - Prompt now strongly prefers early patient-directed contact with
    `requires_response=true` when key facts are missing and the patient can
    speak.
  - Relaxed treatment/diagnostic verbal guidance so the verbal action can
    explain the structured action, reassure, or ask one focused patient history
    question relevant to the current problem.
  - Prompt now warns against diagnosis-specific or medication-heavy treatment
    without enough key context when the patient is stable enough to answer.
  - Existing constraints remain: at most one verbal action, at most one
    structured action, no unstructured extra promised care steps, no selectable
    `no_action`, and no `raw_text`.
- `tests/test_clinician_agent.py`
  - Added prompt assertions for emergency stabilization, direct patient
    questioning, one-question limit, missing key facts, and
    information-dependent treatment context.
  - Added assertions that focused patient questions are permitted alongside
    urgent structured care and are preferred early when missing facts require a
    patient response.

No Orchestrator, StateManager, ActionValidator, TurnLoop, transition engine,
scenario JSON, or non-clinician-agent files were modified.

## Tests Run and Results

```bash
python -m pytest tests/test_clinician_agent.py
```

Result: passed.

```text
36 passed in 0.10s
```

## Exact Next Step

Run a real trajectory on the acute respiratory distress scenario and inspect
whether turn 0 or turn 1 now pairs urgent ABC stabilization with a
patient-directed focused question that activates the patient on the next turn.
