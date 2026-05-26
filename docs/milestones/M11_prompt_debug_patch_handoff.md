# M11 Prompt Debug Patch Handoff

## Patch Section

- Clinician prompt now checks pending/available results before ordering diagnostics.
- Clinician prompt now includes numeric fidelity rule.
- Demo output now prints action params for debugging.
- No clinical hard rules were added.

## Files Modified

- `ed_world_model/agents/clinician.py`
  - Added observation-use guidance to check pending diagnostic results,
    `known_facts.available_results`, and `newly_available_results` before
    ordering diagnostics.
  - Added duplicate diagnostic-order consistency guidance for pending and
    already available results.
  - Added numeric vital fidelity guidance.
- `ed_world_model/orchestration/turn_loop.py`
  - Carries the physiology action through `TurnResult`.
  - Records physiology event payloads with `kind_hint`, `raw_text`, and
    `params`.
- `ed_world_model/orchestration/runner.py`
  - Carries the physiology action through `TrajectoryTurn`.
- `ed_world_model/demo.py`
  - Demo JSON now includes `physiology_action`.
  - Readable trajectory output now prints clinician action type/family/
    kind_hint/params/test_name and physiology action kind_hint/raw_text/params.
- `tests/test_clinician_agent.py`
  - Added prompt assertions for diagnostic duplicate/result-use rules, numeric
    fidelity, and absence of vasopressor-specific or family/kind_hint clinical
    hard rules.
- `tests/test_demo_runner.py`
  - Added readable and JSON output assertions for clinician and physiology
    action params.

No files under `transition_engines/` or `transitions/` were modified by this
patch. The worktree already showed a modified transition JSON before this work;
it was left untouched.

## Tests Run and Results

```bash
python -m pytest tests/test_clinician_agent.py tests/test_demo_runner.py
```

Result: passed.

```text
59 passed in 0.48s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_clinician_agent.py tests/test_demo_runner.py
```

Result: passed.

```text
82 passed in 0.31s
```
