# M14c Patient Emotion Termination Handoff

## Scope

This patch adds the rule that the simulation terminates when any patient
emotion dimension reaches `extreme`.

No scenario JSON, transition engine, adapter, existing test file, or trajectory
output file was modified.

## Changes

- `ed_world_model/state/termination.py`
  - `is_terminated(...)` now returns true when any value in
    `psych_state.patient_emotion.summary()` is `extreme`.
  - `termination_reason(...)` returns `patient_emotion_extreme` for this case.
  - Relative emotion is intentionally not a termination condition.

- `ed_world_model/orchestration/turn_loop.py`
  - After patient emotion is updated from committed messages, the turn loop now
    immediately checks termination.
  - If patient emotion is extreme, the turn returns `terminated=True` with
    `termination_reason="patient_emotion_extreme"`.
  - In this mid-turn termination path, physiology is not called and relative
    emotion is not updated after the patient terminal emotion state.

## Behavior Notes

- If patient emotion is already extreme at the start of a turn, the existing
  start-of-turn termination gate stops before agent generation and physiology.
- If patient emotion becomes extreme during a turn, already committed messages
  and the patient `emotion_update` event remain in the turn result.
- The turn index is not advanced on this terminal mid-turn path, matching the
  existing terminal-start behavior.
- `generate_utterances.py` already breaks on `tt.terminated`, so it will stop
  once the turn loop reports this condition.

## Verification

```bash
python -m py_compile ed_world_model/state/termination.py ed_world_model/orchestration/turn_loop.py
```

Result: passed.

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
37 passed in 0.42s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py::test_termination_prevents_agent_and_physiology_invocation tests/test_full_turn_loop_integration.py::test_noop_emotion_engine_behavior_with_and_without_messages
```

Result: passed.

```text
2 passed in 0.40s
```

Manual behavior checks:

- `GlobalState(psych_state={"patient_emotion": {"fear": "extreme"}})` reports
  `is_terminated=True` and `termination_reason="patient_emotion_extreme"`.
- A turn whose injected emotion engine returns `AgentEmotion(fear="extreme")`
  returns `terminated=True`, `termination_reason="patient_emotion_extreme"`,
  `physiology_called=False`, and records zero physiology adapter calls.
