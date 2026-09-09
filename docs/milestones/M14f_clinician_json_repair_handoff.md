# M14f Clinician JSON Repair Handoff

## Scope

This patch adds a one-shot JSON repair retry inside `ClinicianAgent` so malformed
clinician LLM output does not immediately empty the turn.

No turn-loop behavior, orchestrator behavior, scenario JSON, transition engine,
or existing test file was modified.

## Changes

- `ed_world_model/agents/clinician.py`
  - `ClinicianAgent.generate(...)` now parses clinician Stage 1 decision output
    through `_parse_decision_with_repair(...)`.
  - If Stage 1 parsing raises `ClinicianParserError`, the same clinician LLM
    callable is invoked once with a repair prompt containing:
    - parser error
    - expected JSON shape
    - original task prompt
    - rejected response
  - If the repaired Stage 1 output parses to a meaningful decision, generation
    continues.
  - If repair fails, or repairs to an empty no-op decision
    (`action=null`, no speaking decision), the original parse failure still
    surfaces to the turn loop and is logged as before.
  - Stage 2 final verbal output now has the same one-shot repair path. If the
    repaired final verbal output is `{"verbal_action": null}`, the clinical
    action from Stage 1 is still preserved.
  - Repair prompts truncate long original prompts/rejected outputs to avoid
    oversized repair calls.

## Behavior Notes

- This is an agent-boundary reliability fix. It does not change action
  validation, physiology selection, or fallback `no_action` behavior after both
  original generation and repair fail.
- The retry uses the same injected LLM callable, so tests and runtime clients do
  not need a new dependency or API.
- The repair prompt still forbids markdown fences, commentary, extra keys,
  `raw_text`, chain-of-thought, and internal reasoning.

## Verification

```bash
python -m py_compile ed_world_model/agents/clinician.py
```

Result: passed.

```bash
python -m pytest tests/test_clinician_agent.py
```

Result: passed.

```text
64 passed in 0.16s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py::test_clinician_parser_error_through_runner_is_dropped_and_logged
```

Result: passed.

```text
1 passed in 0.50s
```

```bash
python -m pytest tests/test_agent_prompting.py
```

Result: passed.

```text
12 passed in 0.09s
```

Manual behavior check:

- First clinician decision response: invalid non-JSON text.
- Repair response: valid diagnostic-order decision JSON.
- Result: `ClinicianAgent.generate(...)` made two calls and returned the
  repaired diagnostic action instead of dropping the turn.
