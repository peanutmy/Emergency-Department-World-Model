# M12 Trajectory Logger Handoff

## Files Created or Modified

- `ed_world_model/trajectory/__init__.py`
  - Exposes `TrajectoryLogger`.
- `ed_world_model/trajectory/logger.py`
  - Added lightweight JSON-only `TrajectoryLogger`.
  - Writes `trajectory.json` and `summary.json`.
  - Creates the output directory.
  - Normalizes existing demo/runner turn records without calling agents,
    validators, engines, `StateManager`, or transition engines.
  - Redacts API-key-like metadata keys.
  - Does not implement Markdown output or replay.
- `ed_world_model/orchestration/turn_loop.py`
  - Added `state_before` and `state_after` to `TurnResult`.
  - Captures public snapshots containing only `patient_state` and
    `known_facts`.
- `ed_world_model/orchestration/runner.py`
  - Carries `state_before` and `state_after` through `TrajectoryTurn`.
- `ed_world_model/demo.py`
  - Carries state snapshots through `DemoTurn`.
  - Adds `agent_mode`, `physiology_mode`, `requested_turns`, and `turn_count`
    to `DemoResult.as_dict()`.
  - Adds optional CLI argument `--output-dir`.
  - When `--output-dir` is provided, writes JSON files while preserving
    terminal output behavior.
- `tests/test_trajectory_logger.py`
  - Added focused logger coverage.
- `tests/test_demo_runner.py`
  - Added CLI `--output-dir` coverage in fake mode.

No files under `transition_engines/` or `transitions/` were modified.

## How to Run With `--output-dir`

```bash
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5 \
  --output-dir /tmp/ed_world_model_trajectory
```

Readable terminal output still prints by default. `--json` still controls only
terminal output formatting; it is not required for file output.

## JSON Files Produced

- `trajectory.json`
  - Scenario identifier when available.
  - `agent_mode`, `physiology_mode`, `requested_turns`, `turn_count`, and
    `final_turn_index`.
  - Per-turn records with state snapshots, active agents, committed messages,
    clinician action params, diagnostic orders/releases, nurse shadow
    execution, nurse bedside verbal slots, physiology action
    `kind_hint`/`raw_text`/`params`, validation drops, parser errors, and
    events.
- `summary.json`
  - Scenario identifier when available.
  - `agent_mode`, `physiology_mode`, `requested_turns`, `final_turn_index`,
    `turns_recorded`, validation/parser-error totals, and output timestamp.

No `trajectory.md` or image output is generated.

## State Snapshot Capture

- `state_before` is captured at the start of `TurnLoop.run_turn()`, before
  `release_ready_diagnostic_results(...)` and before agent generation.
- `state_after` is captured after physiology output is applied and before
  `StateManager.advance_turn()` clears per-turn runtime fields.
- The early termination path captures `state_before` before release and
  `state_after` at the termination check after any ready diagnostic release.
- Snapshots include only:
  - `patient_state.vitals`
  - `patient_state.features`
  - `patient_state.status_flags`
  - `known_facts`
- Per-turn snapshots do not include `truth_state`, full `test_bank`, transition
  pairs, source PDFs, API keys, or internal reasoning.

## Tests Run and Results

```bash
python -m pytest tests/test_trajectory_logger.py tests/test_demo_runner.py
```

Result: passed.

```text
25 passed in 0.50s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_trajectory_logger.py tests/test_demo_runner.py
```

Result: passed.

```text
48 passed in 0.30s
```

CLI smoke:

```bash
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 1 \
  --output-dir /tmp/ed_world_model_trajectory_logger_smoke
```

Result: passed and wrote only `trajectory.json` and `summary.json`.

Verification:

```bash
git diff -- transition_engines transitions
```

Result: no diff.

## Known Limitations

- The logger serializes existing trajectory records; it is not a replay engine.
- The logger does not capture observations, raw prompts, raw LLM responses, or
  engine internals beyond fields already present in demo/runner records.
- `save_summary(...)` alone can only write counts provided in metadata.
  `save_all(...)` derives validation/parser-error totals from the trajectory.
- Fallback/mode details are included only when already present in metadata.

## Next Step

Run a real-agent and/or Hybrid physiology demo with `--output-dir` and inspect
the structured JSON for any additional already-available mode/fallback metadata
that should be passed through.
