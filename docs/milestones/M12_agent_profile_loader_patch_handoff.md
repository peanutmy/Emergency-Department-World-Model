# M12 Agent Profile Loader Patch Handoff

## Scope

- `ScenarioLoader` now reads optional top-level `agent_profiles` from scenario
  JSON into `GlobalState.agent_profiles`.
- Missing `agent_profiles` or missing individual roles keep the existing default
  role profiles.
- Supported role keys are the existing `AgentProfiles` fields:
  `clinician`, `nurse`, `patient`, and `relative`.
- Each loaded profile is filtered to the stable `AgentProfile` fields:
  `role`, `name`, and `traits`.
- The containing role key is authoritative. In non-strict mode, a mismatched
  embedded `role` value is corrected to the role key. In strict mode, a mismatch
  raises `ValueError`.
- Malformed individual profile blocks are ignored in non-strict mode and rejected
  in strict mode.
- Demo-created fake and real LLM-backed agents now receive their matching loaded
  role profile via `profile=state.agent_profiles.<role>.model_dump()`.
- The clinician prompt now explicitly uses `AgentProfile` traits to shape only
  communication style, such as tone, concision, bedside manner, explanation
  depth, teaching style, and addressee choice.
- The clinician prompt also states that profile traits must not change observed
  facts, available results, allowed actions, action validation, or grounded
  clinical decision-making.

## Files Changed

- `ed_world_model/scenario_loader.py`
  - Imports `AgentProfile` and `AgentProfiles` from the state model.
  - Adds `_build_agent_profiles(...)`.
  - Passes loaded profiles into `GlobalState(...)`.
- `ed_world_model/demo.py`
  - Passes loaded state profiles into fake and real demo agent constructors.
  - Updates `build_real_demo_agents(...)` to accept `state`.
- `ed_world_model/agents/clinician.py`
  - Adds clinician-specific style-only profile guidance.

No files under `transition_engines/`, `tests/`, `pdf/`, `out/`, or
`docs/v1_3_1.md` were modified by this patch. The working tree contains a
separate user edit under `transitions/Cardiology/Acute Respiratory Distress.json`
that was not modified by this patch.

## Example Scenario Shape

```json
{
  "scenario_description": "...",
  "case_context": {},
  "agent_profiles": {
    "clinician": {
      "role": "clinician",
      "name": "Dr. Lee",
      "traits": {
        "communication_style": "calm, concise, direct",
        "bedside_manner": "reassuring but not falsely certain"
      }
    }
  }
}
```

## Tests Run

```bash
python -m pytest tests/test_scenario_loader.py
```

Result: passed.

```text
28 passed in 0.06s
```

```bash
python -m pytest tests/test_clinician_agent.py tests/test_demo_runner.py tests/test_scenario_loader.py
```

Result: passed.

```text
97 passed in 0.33s
```

```bash
python -m pytest tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
25 passed in 0.24s
```

```bash
python -c "import ed_world_model"
```

Result: passed.

Profile smoke checks:

- Loading a top-level clinician profile populates
  `state.agent_profiles.clinician`.
- Missing nurse profile remains the default nurse profile.
- Strict mode rejects `agent_profiles.clinician.role = "nurse"`.
- A real-mode demo clinician agent prompt includes `Profile:`, the clinician
  profile name, profile traits, and the new style-only clinician profile
  guidance.
- `transitions/Cardiology/Acute Respiratory Distress.json` parses as JSON after
  the separate user edit.

## Next Step

If agents are instantiated outside `ed_world_model.demo`, pass
`state.agent_profiles.<role>.model_dump()` into the concrete agent constructor
until the duplicated state/agent profile schemas are unified.
