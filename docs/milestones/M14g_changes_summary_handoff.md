

###  Patient Emotion Termination

- Added termination when any patient emotion dimension reaches `extreme`.
- Termination reason is `patient_emotion_extreme`.
- Relative emotion does not terminate the simulation.
- If patient emotion becomes extreme during a turn, physiology is not called afterward.

### Patient Speaking More

- Patient still must not invent symptoms, history, allergies, medications, or other facts.
（Stage 1 has stronger guidance:

  - if the latest visible message is addressed to the patient
  - and it asks the patient a question
  - and the patient is alive, conscious, and can speak

  then Stage 1 should usually choose:

  "should_speak": true

  Also, if the patient is directly given treatment instructions while still symptomatic or unstable, Stage 1 may allow a short response
  like:

  Okay... I’m still short of breath.）

- Orchestration — make patient/relative speak when addressed
`ed_world_model/orchestration/orchestrator.py` — new `RECIPIENT_LAST_TURN`
  rule: the patient is activated whenever it was the recipient of any message in
  the previous turn (not only on `requires_response`)

### Clinician History Questions

- Strengthened clinician prompt to ask more focused history questions.
- When patient can speak, clinician should ask focused patient questions about:
  - symptoms, onset, and progression
  - allergies
  - past medical or surgical history
  - home medications
- When patient cannot speak, clinician should ask the relative instead.
- Clinician Stage 2 now reinforces that patient/relative questions should be short and use `requires_response=true`.

### Relative：   patient intubation or Sedation : no utterance and relative start speak;     sedation --
emotion state no change 
- `ed_world_model/orchestration/observation_builder.py` — relative observation carries `relative_emotion`. ：
- keep i don't know: relative observation (family_side_hidden_info=None)？？？

- Emotion accounts for consciousness (freeze when sedated/intubated)
- `ed_world_model/orchestration/turn_loop.py`:
  - `_suppresses_consciousness(action)` → when a **sedation / RSI / intubation /
    surgical-airway** treatment runs, set `is_conscious=False`, `can_speak=False`,
    `features.intubated=True`.
  - Patient emotion is **frozen** (engine not called) while the patient is dead or
    unconscious. The relative (a separate person) keeps updating.



### Clinician JSON Repair

- Added one-shot repair retry inside `ClinicianAgent`.
- If clinician Stage 1 or Stage 2 output is malformed JSON, the agent asks the same LLM once to repair it.
- If repair succeeds, the turn continues.
- If repair fails, the existing `validation_drop` behavior remains.
- Empty no-op repaired Stage 1 output is still treated as failure.


### Random agent traits at scenario load
- **NEW** `ed_world_model/agents/agent_traits.py` — `AgentTraitLibrary`: loads
  per-role trait JSON, samples 1..N random traits + a random level each.


emotion engine:

1. initial emotion : not at all
### Patient Emotion Termination

- Added termination when any patient emotion dimension reaches `extreme`.
- Termination reason is `patient_emotion_extreme`.
- Relative emotion does not terminate the simulation.
- If patient emotion becomes extreme during a turn, physiology is not called afterward.

updated:

1. keep i don't know: relative observation (family_side_hidden_info=None)










## Files Modified

- `generate_utterances.py`
- `ed_world_model/state/termination.py`
- `ed_world_model/orchestration/turn_loop.py`
- `ed_world_model/agents/patient.py`
- `ed_world_model/agents/clinician.py`

## Handoff Files Added

- `docs/milestones/M14b_generate_utterances_physiology_handoff.md`
- `docs/milestones/M14c_patient_emotion_termination_handoff.md`
- `docs/milestones/M14d_patient_stage1_speaking_handoff.md`
- `docs/milestones/M14e_clinician_history_questions_handoff.md`
- `docs/milestones/M14f_clinician_json_repair_handoff.md`
- `docs/milestones/M14g_changes_summary_handoff.md`

## Notes

- Scenario JSON files were not intentionally changed by these patches.
- Transition engines were not changed.
- The current remaining major issue is scenario-specific action availability, especially preventing BiPAP when the scenario says BiPAP is unavailable.
