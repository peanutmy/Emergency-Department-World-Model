# M13 Handoff

## Scope

M13 adds two-stage verbal generation for the patient and clinician agents.

- Stage 1 produces a transient `VerbalDecision`.
- Stage 2 uses the `VerbalDecision` plus role profile/traits and emotion
  context where applicable to produce final `verbal_action` JSON.
- Nurse and relative agents remain unchanged.

## Files Changed

- `ed_world_model/agents/schemas.py`
  - Added `VerbalDecision` with:
    - `speaker`
    - `should_speak`
    - `target`
    - `intent`
    - `reasoning_summary`
    - `key_points`
    - `forbidden_points`
    - `requires_response`
  - No `intent_type`, `raw_text`, `chain_of_thought`,
    `internal_reasoning`, `action`, `profile`, `traits`, or
    `emotion_context` fields were added.
- `ed_world_model/agents/verbal_decision.py`
  - Added strict JSON parser for `VerbalDecision`.
  - Rejects `raw_text`, `chain_of_thought`, `internal_reasoning`, `action`,
    and `intent_type` anywhere in decision JSON.
  - Rejects speaker mismatches and invalid silence shape.
- `ed_world_model/agents/patient.py`
  - Added `build_patient_decision_prompt(...)`.
  - Added `build_patient_verbal_prompt(...)`.
  - `PatientAgent.generate(...)` now runs Stage 1 decision generation before
    final verbal generation.
  - If `should_speak=false`, returns `AgentProposal(verbal_action=None,
    action=None)` without a Stage 2 call.
  - Patient remains verbal-only; action is always `None`.
- `ed_world_model/agents/clinician.py`
  - Added `build_clinician_decision_prompt(...)`.
  - Added `build_clinician_verbal_prompt(...)`.
  - Added `parse_clinician_decision_response(...)`.
  - Added `parse_clinician_verbal_response(...)`.
  - `ClinicianAgent.generate(...)` now runs Stage 1 action plus
    `VerbalDecision`, then Stage 2 final verbal generation only when needed.
  - Action parsing remains structural; `ActionValidator` still owns downstream
    action validation.
- `ed_world_model/orchestration/turn_loop.py`
  - Added optional `agent_verbal_decisions` debug output to `TurnResult`.
  - Debug decisions are collected from agent transient state only after
    generation.
- `ed_world_model/orchestration/runner.py`
  - Carries `agent_verbal_decisions` through `TrajectoryTurn`.
  - Updated `ScriptedLLMCallable` clinician/patient defaults for Stage 1
    parseable silence.
- `ed_world_model/demo.py`
  - Updated fake demo scripts to provide Stage 1 decision JSON and Stage 2
    final verbal JSON.
  - Carries `agent_verbal_decisions` through `DemoTurn`.
- `ed_world_model/trajectory/logger.py`
  - Includes optional `agent_verbal_decisions` in `trajectory.json`.
  - Normalizes only short structured fields:
    `speaker`, `target`, `should_speak`, `intent`, `reasoning_summary`,
    `key_points`, `forbidden_points`, and `requires_response`.
- Focused tests updated in:
  - `tests/test_agent_schemas.py`
  - `tests/test_agent_prompting.py`
  - `tests/test_patient_agent.py`
  - `tests/test_clinician_agent.py`
  - `tests/test_full_turn_loop_integration.py`
  - `tests/test_demo_runner.py`

No files under `transition_engines/` or `transitions/` were modified.

## Design Notes

- Stage 1 prompts explicitly exclude AgentProfile, traits, communication
  style, personality, and emotion context.
- Stage 1 patient prompt sanitizes profile/trait/emotion fields out of the
  rendered observation.
- Stage 1 clinician prompt sanitizes profile/trait/emotion fields, hidden
  `patient_internal_state`, and `truth_state` out of the rendered observation.
- Stage 2 prompts use profile/traits/emotion only to shape wording and tone.
- Stage 2 prompts tell the model to use `VerbalDecision` and the selected
  action as the only plan.
- `VerbalDecision` is transient:
  - not stored in `GlobalState`
  - not appended to `runtime_state.messages`
  - not added to `known_facts`
  - not used by `ActionValidator`
- Only final `verbal_action.content` becomes conversation.
- No behavior actions, nurse physical actions, task queues, new memory systems,
  clinical guidance libraries, or semantic duplicate suppression were added.

## Tests Run

Focused M13 run:

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_patient_agent.py tests/test_clinician_agent.py
```

Result: passed.

```text
138 passed in 0.14s
```

Full focused plus adjacent run:

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_patient_agent.py tests/test_clinician_agent.py tests/test_turn_loop_stub.py tests/test_full_turn_loop_integration.py tests/test_demo_runner.py tests/test_trajectory_logger.py
```

Result: passed.

```text
224 passed in 0.43s
```

Additional checks:

```bash
python -m pytest tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
26 passed in 0.26s
```

```bash
python -m pytest tests/test_turn_loop_stub.py
```

Result: passed.

```text
31 passed in 0.08s
```

```bash
python -m pytest tests/test_demo_runner.py tests/test_trajectory_logger.py
```

Result: passed.

```text
29 passed in 0.30s
```

```bash
git diff -- transition_engines transitions
```

Result: no diff.

## Known Limitations

- Verbal/action semantic consistency remains mostly prompt-level. The
  clinician Stage 1 parser ensures only one structured action shape, but it
  does not semantically inspect free text in `key_points`.
- Patient truthfulness is improved through a structured decision and stricter
  prompts, but no deterministic medical-fact extraction or semantic
  contradiction checker was added.
- `agent_verbal_decisions` are debug output only. They are not replay logic.
- Real LLM output still depends on prompt following; no provider-side JSON
  schema, retry, repair, or tool calling was added.

## Next Step

Run a real-agent trajectory with `--output-dir` and inspect whether Stage 1
decisions reduce invented patient symptoms and clinician extra-action verbal
claims without harming turn-loop behavior.
