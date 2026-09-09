# M0 Handoff

## Files/Directories Created

- `AGENTS.md`
- `docs/milestones/`
- `docs/milestones/README.md`
- `docs/milestones/M0_handoff.md`
- `ed_world_model/`
- `ed_world_model/__init__.py`
- `ed_world_model/state/`
- `ed_world_model/state/__init__.py`
- `ed_world_model/actions/`
- `ed_world_model/actions/__init__.py`
- `ed_world_model/adapters/`
- `ed_world_model/adapters/__init__.py`
- `ed_world_model/orchestration/`
- `ed_world_model/orchestration/__init__.py`
- `ed_world_model/agents/`
- `ed_world_model/agents/__init__.py`

All `__init__.py` files are intentionally empty. No business-logic modules were created.

## Files Intentionally Not Touched

During original M0, these files/directories were intentionally not touched:

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- existing scenario JSON files
- `tests/`
- `pdf/`
- `out/`
- existing engine code
- existing test code

Git status after M0 also reports `D 1.py`. That file was already absent during M0 inspection, and M0 did not restore, recreate, or modify it.

## Repo Structure Observed

Observed top-level structure during M0:

```text
.
├── .agents/
├── .codex/
├── .git/
├── .pytest_cache/
├── __pycache__/
├── docs/
├── out/
├── pdf/
├── tests/
├── transition_engines/
└── transitions/
```

`docs/v1_3_1.md` already existed before M0 changes. No top-level `pyproject.toml`, `requirements.txt`, `setup.cfg`, `pytest.ini`, or `Makefile` was observed.

Existing transition engine code appears to live in `transition_engines/`. Existing transition-pair scenario data appears to live in `transitions/`. Existing tests appear to live in `tests/`.

## Commands/Checks Run

```bash
python -c "import ed_world_model"
```

Result: passed.

```bash
wc -c ed_world_model/__init__.py ed_world_model/state/__init__.py ed_world_model/actions/__init__.py ed_world_model/adapters/__init__.py ed_world_model/orchestration/__init__.py ed_world_model/agents/__init__.py
```

Result: all package marker files are `0` bytes.

Broad pytest was not run because M0 only created documentation and empty package marker files, and no top-level pytest configuration was observed.

The import check generated `ed_world_model/__pycache__/`; it was removed after the check so the final M0 file set contains only the approved files/directories.

## Uncertainty or Mismatch with `docs/v1_3_1.md`

No mismatch was identified during M0. Implementation of the architecture described in `docs/v1_3_1.md` is intentionally deferred to later milestones.

## Known Limitations

- `ed_world_model/` is only an empty package skeleton.
- No state models, action registry, validators, adapters, orchestration, agents, or turn loop exist yet.
- Existing transition engines are not wrapped yet.
- The external/pluggable emotion engine interface and `NoopEmotionEngine` fallback are not implemented yet.

## Exact Next Step

M1a GlobalState models.

## M0_patch Note

M0_patch made documentation-only updates to `docs/v1_3_1.md`, `AGENTS.md`, and this handoff. The diagnostic `test_bank` rule now says items contain `name`, `result`, and optional `turnaround_turns`; runtime reads turnaround from the ordered test's `test_bank` item and falls back to `DEFAULT_DIAGNOSTIC_TURNAROUND_TURNS = 2` when missing.
