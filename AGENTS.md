# Coding Agent Guide

## Project Goal

The ED Multi-Agent World Model v1.3.1 is a lightweight runtime model for an emergency department simulation. It will coordinate state, clinical actions, observations, agent turns, physiology transitions, and patient emotion while preserving strict separation between released clinical facts and hidden/source data.

This repository already contains tested transition-pair engines and scenario data. New world-model code must wrap or use those systems through adapter layers without changing existing engine or scenario files unless explicitly instructed.

## Current Milestone

Current milestone: M0.

M0 is limited to repository guidance, milestone handoff convention, and an empty package skeleton. Do not implement GlobalState, StateManager, action validation, orchestration, observation building, adapters, agents, or the turn loop in M0.

## Repository Structure

Observed top-level structure:

```text
.
├── docs/
│   ├── v1_3_1.md
│   └── milestones/
├── ed_world_model/
│   ├── state/
│   ├── actions/
│   ├── adapters/
│   ├── orchestration/
│   └── agents/
├── transition_engines/
├── transitions/
├── tests/
├── pdf/
└── out/
```

Existing transition engine code lives in `transition_engines/`. Existing scenario JSON data lives in `transitions/`. Existing tests live in `tests/`. Source PDFs live in `pdf/`. Generated evaluation outputs live in `out/`.

## Code Locations

New world-model code lives under `ed_world_model/`.

Runtime adapter layers belong under `ed_world_model/adapters/`. Later milestones may add physiology and emotion adapters here; do not create those files until the relevant milestone asks for them.

Use `ed_world_model/adapters/`, not `ed_world_model/engines/`, to avoid confusion with the existing `transition_engines/` package.

## Do Not Touch Unless Explicitly Instructed

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `tests/`
- `pdf/`
- `out/`
- existing engine code
- existing test code

## Checks

No top-level `pyproject.toml`, `requirements.txt`, `setup.cfg`, `pytest.ini`, or `Makefile` was observed during M0 inspection.

Lightweight import check:

```bash
python -c "import ed_world_model"
```

Existing test suite, when explicitly needed:

```bash
python -m pytest tests
```

For broad pytest runs, consider cost and scope first. If a milestone only creates docs or empty package files, prefer the lightweight import check.

## Core Architecture Rules

- New world-model code lives under `ed_world_model/`.
- Existing `transition_engines/` and `transitions/` are tested/existing code; do not modify them unless explicitly instructed.
- Existing scenario JSON files should not be modified.
- `StateManager` will be the only writer to `GlobalState`.
- `known_facts` is the only clinical-team factual store.
- Memory is rolling conversation window only.
- Runtime `raw_text` is always null.
- Diagnostic `test_bank` items contain `name`, `result`, and optional `turnaround_turns`.
- Diagnostic turnaround is read from the ordered test's `test_bank` item; if `turnaround_turns` is missing, use `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2`.
- No test bundles in v1.3.1.
- Clinician may produce at most one action per turn.
- Nurse has no physical action in v1.3.1.
- Valid clinician treatment orders are shadow-executed by the nurse role.
- Nurse bedside verbal slot is reactive and not selected by the Orchestrator.
- Physiology engines must not receive unreleased diagnostic results.
- Emotion engine only uses conversation input, not vitals.
- If no valid medical treatment order exists in a turn, the physiology path uses `no_action`.
- `raw_text` must remain null even for `no_action`.
- The emotion engine is external/pluggable. v1.3.1 will later use an adapter interface and a `NoopEmotionEngine` fallback.
- Do not add task queues, async workflows, retries, plugin frameworks, or extra managers unless a milestone explicitly asks for them.

## Milestone Handoff Rule

Every coding session must end by writing `docs/milestones/<milestone>_handoff.md`.

New Codex sessions should read this file and the previous milestone handoff before writing code.
