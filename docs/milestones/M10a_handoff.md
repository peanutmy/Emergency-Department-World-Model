# M10a Handoff

## Files Created or Modified

- `ed_world_model/demo.py`
  - Added a reusable controlled demo runner:
    - `run_demo_scenario(...)`
    - `build_fake_demo_agents(...)`
    - `render_readable_trajectory(...)`
    - `render_demo_json(...)`
    - `main(...)`
  - Loads scenario JSON through `ScenarioLoader`.
  - Runs turns through the existing M9 `IntegrationRunner` / M6 `TurnLoop`.
  - Uses deterministic fake LLM callables by default:
    - clinician orders the first available diagnostic test on turn 0 when one
      exists,
    - clinician is silent/no-action on turn 1,
    - clinician orders oxygen support on turn 2,
    - nurse, patient, and relative default to parseable silence.
  - Uses `RecordingStubPhysiologyAdapter` and `NoopEmotionEngine` by default.
  - Captures per-turn patient-state summaries after physiology update.
  - Renders readable and JSON trajectories.
- `examples/run_demo_scenario.py`
  - Added a small CLI entrypoint delegating to `ed_world_model.demo.main`.
  - Supports:
    - `--scenario PATH`
    - `--turns INT`
    - `--mode fake`
    - `--json`
- `tests/test_demo_runner.py`
  - Added focused M10a coverage for scenario loading, no API dependency,
    readable output, JSON output, source-file immutability, `ScenarioLoader`
    usage, no direct transition-engine import, diagnostic release after
    turnaround, treatment/no-action physiology kind hints, CLI entrypoint
    behavior, and verbal-only non-clinician demo agents.
- `docs/milestones/M10a_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code
- existing scenario/evaluation output files

No real LLM client, prompt repair/retry loop, semantic duplicate suppression,
behavior action, nurse physical action, relative emotion transition, scenario
JSON mutation, task queue, async workflow, PK/PD, or engine routing was added.

## Tests Run and Results

```bash
python -m pytest tests/test_demo_runner.py
```

Result: passed.

```text
8 passed in 0.13s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_demo_runner.py
```

Result: passed.

```text
29 passed in 0.10s
```

CLI smoke run:

```bash
python examples/run_demo_scenario.py --scenario tests/fixtures/scenario_loader_minimal.json --turns 2
```

Result: passed and printed a readable two-turn trajectory.

Broad pytest was not run because M10a only adds the demo wrapper, CLI script,
and focused tests. Prior handoffs document unrelated broader-suite
`PatientEmotion.notes` failures.

## How to Run the Demo

Readable output:

```bash
python examples/run_demo_scenario.py --scenario tests/fixtures/scenario_loader_minimal.json --turns 5
```

JSON output:

```bash
python examples/run_demo_scenario.py --scenario tests/fixtures/scenario_loader_minimal.json --turns 5 --json
```

The command does not require `OPENAI_API_KEY` or any external service
configuration.

## Sample Command

```bash
python examples/run_demo_scenario.py --scenario transitions/Cardiology/Acute\ Respiratory\ Distress.json --turns 5
```

Use any extracted scenario JSON path that `ScenarioLoader` can load. The demo
does not mutate the source scenario file.

## Known Limitations

- The fake clinician script is deterministic and not clinically intelligent.
- The oxygen order on turn 2 is scripted; it is not inferred from diagnostic
  results or patient condition.
- Nurse, patient, and relative default to silence unless the demo script is
  extended.
- The stub physiology adapter records the engine-facing action and returns a
  no-op vitals update by default; it does not model disease progression.
- JSON/readable trajectories expose compact runtime debug records, not a full
  `TrajectoryLogger`.
- `--mode` supports only `fake` in M10a.

## Exact Next Step

M10b real LLM client wrapper with fake/real switch.

## M10a_patch Controlled Demo Visibility

### Files Modified

- `ed_world_model/demo.py`
  - Updated the fake script so the default demo visibly exercises:
    - clinician question requiring a patient response,
    - patient required-response conversation on the following turn,
    - nurse result reporting when diagnostic results become available,
    - nurse bedside verbal slot with `spoke=True`,
    - a later nurse bedside verbal slot with `spoke=False`,
    - fake physiology effects for `no_action` and `oxygen_support`.
  - Added readable and JSON `nurse_bedside_slots` output.
  - Replaced the no-op demo physiology with a deterministic fake physiology
    factory:
    - `oxygen_support` updates `features.oxygen_device` and `features.FiO2`
      from action params.
    - `oxygen_support` increases `O2Sat` by 5 up to 95 when `O2Sat` exists
      and is below 95.
    - `no_action` keeps vitals stable by default.
    - `no_action` decreases `O2Sat` by 1 when `O2Sat < 90`.
- `ed_world_model/state/state_manager.py`
  - Added the minimal pending-question hooks already expected by the M6
    `TurnLoop`:
    - `create_pending_question(...)`
    - `resolve_pending_questions_for_agent(...)`
  - These hooks keep `StateManager` as the writer for
    `runtime_state.pending_questions` and
    `runtime_state.required_response_agents`.
- `tests/test_demo_runner.py`
  - Expanded focused M10a coverage from 8 to 11 tests.
  - Added assertions for required-response patient activation, nurse result
    reporting, bedside slot spoken/silent payloads, and low-O2 fake physiology
    effects.
- `docs/milestones/M10a_handoff.md`
  - Added this patch section.

### Files Intentionally Not Touched

- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code

No real LLM client, external API call, real clinical reasoning, prompt repair,
retry loop, semantic duplicate suppression, behavior action, nurse physical
action, task queue, async workflow, PK/PD, or engine routing was added.

### Tests Run and Results

```bash
python -m pytest tests/test_demo_runner.py
```

Result: passed.

```text
11 passed in 0.12s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_demo_runner.py
```

Result: passed.

```text
32 passed in 0.12s
```

CLI smoke run:

```bash
python examples/run_demo_scenario.py --scenario tests/fixtures/scenario_loader_minimal.json --turns 4
```

Result: passed and printed a four-turn trajectory showing patient response,
nurse result reporting, `nurse_bedside_verbal_slot triggered=True spoke=True`,
and `nurse_bedside_verbal_slot triggered=True spoke=False`.

### Known Limitations

- The fake clinician script remains deterministic and not clinically
  intelligent.
- The repeated oxygen order exists only to visibly exercise both spoken and
  silent bedside-slot payloads.
- Fake physiology remains simple deterministic demo behavior, not a real
  transition engine.

### Exact Next Step

M10b real LLM client wrapper with fake/real switch.
