# M11 Grounding Patch Handoff

## Patch Section

- Clinician prompt now grounds diagnostic-result references to exact observation
  contents.
- Clinician prompt now includes numeric/rationale grounding.
- No clinical action-specific hard rules were added.
- Vent/ventilation field semantics were intentionally not changed in this patch.

## Files Modified

- `ed_world_model/agents/clinician.py`
  - Expanded diagnostic-result prompt grounding so referenced results must use
    exact test names/results present in the current observation.
  - Added prompt-only constraints against inferring or substituting a different
    test name.
  - Clarified pending tests must be described as pending/in progress, not
    available.
  - Clarified unavailable results must not be described as ready/available.
  - Expanded numeric/rationale grounding so verbal reasons must match observed
    vitals, known facts, pending results, and available results.
- `tests/test_clinician_agent.py`
  - Added focused assertions for diagnostic-result grounding, exact test-name
    use, pending-vs-available wording, duplicate pending/available test
    avoidance, exact vital values, unsupported rationale prevention, and absence
    of scenario-specific mismatch examples or treatment-specific hard rules.
- `docs/milestones/M11_grounding_patch_handoff.md`
  - Added this handoff.

No demo/debug output code was changed by this patch; the existing action-params
debug output was kept. No files under `transition_engines/` or `transitions/`
were modified by this patch. The worktree already showed a modified transition
JSON before this work; it was left untouched.

## Tests Run and Results

```bash
python -m pytest tests/test_clinician_agent.py tests/test_demo_runner.py
```

Result: passed.

```text
67 passed in 0.33s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py tests/test_clinician_agent.py tests/test_demo_runner.py
```

Result: passed.

```text
90 passed in 0.32s
```
