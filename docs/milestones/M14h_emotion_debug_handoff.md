# M14h Emotion Debug Handoff

## Scope

Added case-scoped emotion debug output for the intra-abdominal sepsis run only.

This does not change `TurnLoop`, `AgentEmotion`, the emotion engine adapter,
scenario JSON, transition engines, or tests.

## Changes

- `generate_utterances.py`
  - Added an intra-abdominal sepsis case detector based on the scenario path or
    scenario description.
  - For that case only, each output turn now includes
    `patient_emotion_last_error`.
  - If the patient emotion adapter reports a non-empty `last_error` on a turn
    where a patient emotion update was recorded, the saved JSON turn record also
    appends an `emotion_engine_error` event:

```json
{
  "type": "emotion_engine_error",
  "turn_index": 3,
  "payload": {
    "agent": "patient_emotion",
    "error": "emotion step failed: ..."
  }
}
```

## Behavior Notes

- Non-intra-abdominal-sepsis runs are unchanged and do not get the extra debug
  field or synthetic event.
- The debug event is written only to the JSON record produced by
  `generate_utterances.py`; it is not inserted into runtime state.
- A null `patient_emotion_last_error` means the adapter did not report an error
  for that turn. It does not prove that the model returned useful reasoning; an
  empty-but-valid `reasoning` string can still produce `notes=null`.

## Verification

```bash
python -m py_compile generate_utterances.py
```

Result: passed.

```bash
python -c "import generate_utterances as g; print(g._is_intra_abdominal_sepsis_case('transitions/Resuscitation/Intra-abdominal Sepsis.json', '')); print(g._is_intra_abdominal_sepsis_case('transitions/Cardiology/Acute Respiratory Distress.json', ''))"
```

Result:

```text
True
False
```

```bash
python -c "import generate_utterances as g; E=type('E', (), {'last_error':'boom'}); r={'events':[]}; g._add_case_scoped_emotion_debug(r, turn_index=3, emotion_engine=E(), emotion_update_recorded=True); print(r)"
```

Result: wrote `patient_emotion_last_error` and an `emotion_engine_error` event.
