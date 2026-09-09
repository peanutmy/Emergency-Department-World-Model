# M8b Handoff

## Files Created or Modified

- `ed_world_model/agents/clinician.py`
  - Added `ClinicianAgent` with injected `llm_callable(prompt) -> str`.
  - Added `build_clinician_prompt(...)` for clinician-only prompt construction.
  - Added `parse_clinician_response(...)` /
    `parse_clinician_proposal(...)` for strict JSON parsing into the existing
    `ClinicianProposal = AgentProposal` schema.
  - Added `ClinicianParserError` for clear parser failures.
- `tests/test_clinician_agent.py`
  - Added focused M8b prompt, parser, fake-callable generation, import-boundary,
    and optional ActionValidator integration coverage.
- `docs/milestones/M8b_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `transition_engines/`
- `transitions/`
- scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code
- turn-loop code
- patient, nurse, and relative agent implementations

No real LLM API client, retry/repair loop, clinical scoring, scenario-loader
change, or turn-loop integration was added.

## Tests Run and Results

```bash
python -m pytest tests/test_clinician_agent.py
```

Result: passed.

```text
24 passed in 0.11s
```

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_action_registry.py tests/test_action_validator.py tests/test_clinician_agent.py
```

Result: passed.

```text
99 passed in 0.11s
```

```bash
python -c "import ed_world_model; import ed_world_model.agents.clinician"
```

Result: passed.

Broad pytest was not run because M8b only adds the clinician agent prompt/parser
boundary and the focused plus adjacent agent/action suites passed.

## Clinician Prompt and Parser Behavior

- Prompt construction uses only:
  - partial clinician observation,
  - optional `AgentProfile`,
  - `ActionRegistry` clinician-selectable family/`kind_hint` menu,
  - available diagnostic test names from the observation or explicit argument.
- The prompt says the clinician receives only a partial role-specific
  observation, must not assume hidden state, may produce at most one
  `verbal_action` and one `action`, and may set `action` to `null`.
- The prompt says `no_action` is not selectable, `raw_text` must not be
  included, internal reasoning must not be included, and chain-of-thought must
  not be included.
- The action menu excludes non-clinician-selectable registry entries such as
  `no_action`.
- The parser accepts strict JSON only, rejects duplicate JSON keys and JSON
  constants such as `NaN`, and requires the top-level `verbal_action` and
  `action` fields.
- The parser rejects `raw_text`, `internal_reasoning`, and `chain_of_thought`
  anywhere in the response.
- The parser rejects explicit `no_action`, list-form actions, top-level action
  aliases, extra top-level fields, extra diagnostic-order fields, extra
  medical-treatment-order fields, and malformed verbal/action shapes.
- The parser preserves `params: null`; it does not fill missing clinical values.
- Diagnostic actions normalize to exactly `{"type": "diagnostic_order",
  "test_name": ...}` and reject invented result fields.
- The parser does not validate medical correctness, known `family` /
  `kind_hint`, or diagnostic test availability. `ActionValidator` remains
  responsible for that later structural validation.
- `ClinicianAgent.generate(...)` calls only the injected fake/runtime callable
  and returns the parsed `ClinicianProposal`.

## Deviations From `docs/v1_3_1.md` and Why

- No intentional architecture deviation was introduced.
- The user-facing expected JSON shape uses `verbal_action.target`, while the
  existing M8a/M6 shared `VerbalAction` schema uses `recipient`. M8b accepts
  `target` at the clinician LLM boundary and normalizes it to `recipient` before
  returning the shared `AgentProposal`. This avoids creating an incompatible
  duplicate proposal model.
- The parser is deliberately stricter than the generic M8a parser for clinician
  LLM output shape so unexpected fields are rejected rather than silently stored
  in final proposals.

## Known Limitations

- No real clinician LLM client is implemented.
- No patient, nurse, or relative LLM agents are implemented.
- No retry, repair, or markdown extraction path exists; malformed output fails
  clearly.
- Prompt quality does not guarantee clinical quality.
- The parser does not call `ActionValidator` during normal parsing.
- There is no turn-loop integration for this clinician LLM agent yet.

## Exact Next Step

M8c Patient/Nurse/Relative verbal agent parsers.

## M8b_patch

- Updated `ed_world_model/agents/clinician.py::build_clinician_prompt(...)` so
  the clinician is instructed to silently reason before final JSON output:
  - if producing a `verbal_action`, what the clinician would say and who they
    would address based on the observation, recent messages, and clinician
    profile/traits;
  - if producing a clinician action, which single `medical_treatment_order` or
    `diagnostic_order` fits the observation, released results, action menu, and
    available diagnostic test names.
- The prompt still forbids outputting reasoning, rationale, analysis,
  `internal_reasoning`, or chain-of-thought, and the parser still rejects
  forbidden reasoning fields anywhere.
- Added focused prompt assertions in `tests/test_clinician_agent.py`.

Patch test run:

```bash
python -m pytest tests/test_clinician_agent.py
```

Result: passed.

```text
24 passed in 0.08s
```
