# ED Multi-Agent World Model

This repository is building a lightweight runtime model for emergency
department simulations. The goal is to let several role-based agents share a
turn-by-turn clinical scene while keeping a clear boundary between hidden case
truth and what the care team has actually observed.

The project treats the simulation as a small world model rather than a single
chat prompt. A central state object tracks patient physiology, released clinical
facts, conversation, diagnostic timing, and runtime events. Agents only receive
role-specific observations derived from that state, so a clinician can see
released facts and current patient status, while a patient can see patient-owned
internal context, and hidden diagnostic results stay hidden until released.

## Design Idea

The architecture is intentionally simple:

- `GlobalState` stores the world.
- `StateManager` is the mutation boundary.
- `Orchestrator` decides which agents are active at the start of a turn.
- `ObservationBuilder` creates filtered views for those active agents.
- `ActionValidator` checks clinician orders before anything is executed.
- `TurnLoop` coordinates one simulation turn.
- Adapters connect runtime actions to physiology or emotion engines.

The clinician may order diagnostics or treatments. Diagnostic orders create
pending results and do not affect physiology. Valid treatment orders are
recorded as nurse shadow execution events, then passed to the physiology path.
If there is no valid treatment, the physiology path receives a one-minute
`no_action` update.

The nurse, patient, and relative are verbal-only in this version. The nurse can
also get a reactive bedside verbal slot after a valid treatment order, but that
slot is not selected by the orchestrator.

## Current Runtime Path

The integration runner wires the existing pieces together through injected
agent callables and a physiology adapter. The same path can be exercised with
scripted agents for repeatable tests or with LLM-backed agents for live
trajectory experiments.

Example components:

- scenario loading through `ScenarioLoader`
- turn execution through `TurnLoop`
- default no-op emotion through `NoopEmotionEngine`
- scripted or LLM-backed agent outputs for tests and demos
- recorded physiology calls for debugging

## Repository Layout

```text
ed_world_model/
  actions/          action taxonomy and validation
  adapters/         physiology and emotion adapter boundaries
  agents/           role prompt/parser and scripted-agent helpers
  orchestration/    orchestrator, observations, turn loop, runner
  state/            GlobalState, StateManager, termination helpers

transition_engines/ existing transition-pair engines
transitions/        existing extracted scenario data
tests/              focused unit and integration tests
docs/               design notes and milestone handoffs
```

Existing transition engines and scenario JSON files are treated as source data
and compatibility surfaces. New world-model runtime code lives under
`ed_world_model/`.

## Useful Checks

Run the current full-turn integration tests:

```bash
python -m pytest tests/test_full_turn_loop_integration.py
```

Run the adjacent turn-loop checks:

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_full_turn_loop_integration.py
```

Lightweight import check:

```bash
python -c "import ed_world_model; import ed_world_model.orchestration.runner"
```

## Status

The project supports integration trajectories through a pluggable agent
boundary. The main design focus is the runtime boundary between state,
observations, validated actions, and engine adapters.
