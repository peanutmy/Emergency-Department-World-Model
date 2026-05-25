# M9 Handoff

## Files Created or Modified

- `ed_world_model/orchestration/runner.py`
  - Added `IntegrationRunner`, a thin multi-turn wrapper around the existing
    M6 `TurnLoop`.
  - Added `TrajectoryTurn`, a compact debug record with active agents,
    committed messages, validation results, physiology action kind, released
    diagnostics, events, termination status, and turn indexes.
  - Added `RecordingStubPhysiologyAdapter`, a deterministic fake physiology
    adapter that records received engine-facing actions and returns a no-op
    vitals update by default.
  - Added `ScriptedLLMCallable`, a fake callable for deterministic strict-JSON
    LLM-agent tests. It records prompts and never calls external services.
  - Added helper functions:
    - `run_trajectory(...)`
    - `default_scripted_agents()`
    - `scripted_agents(...)`
  - Runner accepts `GlobalState`, a scenario JSON path, a scenario mapping, or
    `None`. Scenario paths/mappings are loaded through `ScenarioLoader`.
  - Runner injects existing components: `TurnLoop`, `Orchestrator`,
    `ObservationBuilder`, `ActionValidator`, `StateManager`, and
    `NoopEmotionEngine`.
- `tests/test_full_turn_loop_integration.py`
  - Added 15 focused M9 integration tests covering one-turn and multi-turn
    trajectories, scenario path loading, diagnostics, treatment orders, nurse
    shadow execution, nurse bedside verbal slot, verbal-only agents, invalid
    actions, fake callable usage, hidden-result leakage boundaries, partial
    observations, emotion behavior, termination, event promotion, and trajectory
    debug fields.
- `docs/milestones/M9_handoff.md`
  - Added this handoff.
- `README.md`
  - Added a concise top-level project README explaining the design idea,
    runtime boundaries, major modules, repository layout, useful checks, and
    current status.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code
- existing test files other than adding the new M9 test file

`examples/run_fake_scenario.py` was not added because no `examples/` directory
currently exists and the integration target is covered by the runner and focused
tests.

The worktree already contained pending M8c agent files and tests before M9.
M9 did not revert or rewrite that work.

## Tests Run and Results

```bash
python -m pytest tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
15 passed in 0.07s
```

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
32 passed in 0.09s
```

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_clinician_agent.py tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
134 passed in 0.20s
```

```bash
python -c "import ed_world_model; import ed_world_model.orchestration.runner"
```

Result: passed.

Broad pytest was not run. Prior handoffs document unrelated broader-suite
`PatientEmotion.notes` failures, and M9 scope was integration runner plus
focused trajectory tests.

## Integration Behavior

- `IntegrationRunner` constructs a `StateManager` and wires it into the existing
  `TurnLoop`.
- If initialized with a scenario path or mapping, the runner uses
  `ScenarioLoader` and ignores transition pairs through the existing loader
  boundary.
- Default agents are deterministic silent stubs for clinician, nurse, patient,
  and relative.
- Tests can inject existing M8 LLM-style agents with `ScriptedLLMCallable`.
  These fake callables return fixed JSON and record prompts only.
- Default physiology is `RecordingStubPhysiologyAdapter`, which records:
  - turn index,
  - engine-facing action,
  - released diagnostic results visible through `known_facts`.
- The fake physiology adapter does not read `truth_state.test_bank`, does not
  call Hybrid/OpenAI, and returns a deterministic no-op update by default.
- `NoopEmotionEngine` remains the default emotion engine. It is invoked only
  when the existing turn loop commits verbal messages.
- The runner returns `list[TrajectoryTurn]`, not full `GlobalState` snapshots,
  so hidden/source state is not exposed through trajectory records.

## Deviations From `docs/v1_3_1.md` and Why

- No intentional architecture deviation was introduced.
- `TrajectoryTurn` is a compact debug object, not a full `TrajectoryLogger`.
  This matches M9's integration-runner scope and avoids adding a new logging
  subsystem.
- The fake physiology adapter is a test/demo stub, not a physiology transition
  engine. It exists so M9 can run full trajectories without API keys or engine
  routing.
- `IntegrationRunner` supports scenario mappings in addition to paths and
  `GlobalState` for test ergonomics.

## Known Limitations

- No prompt repair, retry, API configuration, or external service integration
  exists.
- No real clinical intelligence is added.
- No behavior actions, nurse physical actions, task queues, async workflows,
  memory summarization, semantic duplicate suppression, PK/PD, or engine
  routing were added.
- The runner does not replace `TurnLoop`; it delegates turn semantics to it.
- The fake physiology adapter's default update preserves current vitals and
  does not model disease progression.
- There is still no full configurable `TrajectoryLogger`.

## Exact Next Step

Agent-runtime integration or an end-to-end demo with a controlled deterministic
scenario.

## M9 Documentation Follow-Up

Added `README.md` after the initial M9 implementation to provide a short
high-level explanation of the project design. The README describes the agent
boundary as pluggable, supporting scripted tests or LLM-backed trajectory
experiments. No production code changed in this follow-up, and no additional
tests were run because the update is documentation-only.

## M9_patch Agent Parser Error Handling

### Files Modified

- `ed_world_model/orchestration/turn_loop.py`
  - Primary active-agent generation now catches role parser errors alongside
    `pydantic.ValidationError`.
  - Reactive nurse bedside-slot generation now catches the same generation
    errors.
  - Parser/validation generation failures are logged as `validation_drop`
    events and the affected agent is treated as silent for the turn.
  - Generation-drop event payloads include:
    - `agent`
    - `item`
    - `errors`
    - `phase`
    - `error_type`
    - `error_message`
  - Supported generation-error classes:
    - `ClinicianParserError`
    - `PatientParserError`
    - `NurseParserError`
    - `RelativeParserError`
    - `pydantic.ValidationError`
- `ed_world_model/orchestration/runner.py`
  - `ScriptedLLMCallable` now accepts a `role`.
  - Exhausted clinician scripts default to
    `{"verbal_action": null, "action": null}`.
  - Exhausted patient, nurse, and relative scripts default to
    `{"verbal_action": null}`.
  - Added a docstring note that a provided `GlobalState` is shared with the
    runner and mutated through `StateManager`.
- `tests/test_full_turn_loop_integration.py`
  - Added focused coverage for parser-error-through-runner behavior, verbal-only
    parser errors, role-aware scripted callable defaults, multi-turn
    verbal-only exhaustion, and nurse bedside-slot parser errors.

### Behavior

- Malformed or forbidden agent output no longer crashes `run_turn`.
- No retry, JSON repair, semantic duplicate suppression, behavior action,
  nurse physical action, task queue, async workflow, memory summarization,
  PK/PD, or engine routing was added.
- On parser failure, the turn continues with a silent proposal for that agent.
- If the clinician proposal is dropped and no valid treatment exists, physiology
  receives the existing `no_action` payload.
- If the nurse bedside-slot proposal is dropped after a valid treatment order,
  the treatment physiology path still proceeds.
- Verbal-only parsers remain strict and still reject an `action` key, including
  `action: null`.

### Tests Run and Results

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_turn_loop_stub.py
```

Result: passed.

```text
38 passed in 0.11s
```

```bash
python -m pytest tests/test_clinician_agent.py tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
119 passed in 0.17s
```

Broad pytest was not run because this was a focused M9 patch.

### Exact Next Step

Real LLM wrapper integration or a controlled demo runner.
