# M8a Handoff

## Files Created or Modified

- `ed_world_model/agents/schemas.py`
  - Extended the existing M6 final-output schemas without replacing them.
  - Added `AgentRole`.
  - Added `AgentProfile` for agent-generation context:
    - `role`
    - optional `name`
    - `traits: dict[str, str | int | float | bool | list[str] | None]`
  - Added `AgentRuntimeInput` for future LLM-agent inputs:
    - `role`
    - role-specific partial `observation`
    - optional `profile`
    - optional `emotion_context`
    - optional `recent_messages`
    - optional `turn_index`
  - Kept `VerbalAction` with existing M6 `recipient` semantics and no
    `target` field.
  - Kept `AgentProposal` as final output with optional `verbal_action` and
    optional `action`.
  - Added `ClinicianProposal = AgentProposal`.
  - Added `VerbalOnlyProposal` for nurse/patient/relative final-output shape.
- `ed_world_model/agents/prompting.py`
  - Added deterministic prompt builders:
    - `build_agent_prompt`
    - `build_clinician_prompt`
    - `build_nurse_prompt`
    - `build_patient_prompt`
    - `build_relative_prompt`
  - Added `parse_agent_proposal(...)`, a generic JSON-to-`AgentProposal`
    parser.
- `tests/test_agent_schemas.py`
  - Added focused schema coverage.
- `tests/test_agent_prompting.py`
  - Added focused prompt and generic parser coverage.
- `docs/milestones/M8a_handoff.md`
  - Added this handoff.

## Files Intentionally Not Touched

- `docs/v1_3_1.md`
- `StateManager`
- `Orchestrator`
- `ObservationBuilder`
- turn-loop code
- `transition_engines/`
- `transitions/`
- scenario JSON files
- `pdf/`
- `out/`
- existing transition engine code

No real clinician, nurse, patient, or relative LLM agents were implemented.
No OpenAI or external API calls were added.

## Tests Run and Results

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
21 passed in 0.06s
```

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
38 passed in 0.08s
```

Manual import check:

```bash
python -c "import ed_world_model; import ed_world_model.agents.prompting"
```

Result: passed.

Broad pytest was not run because M8a only adds shared agent schema/prompt
helpers and the requested focused plus adjacent turn-loop suites passed.

## Schema and Prompt Helper Behavior

- `AgentRuntimeInput` represents only the agent-facing generation input. It
  does not include full `GlobalState`.
- `observation` is treated as already partial and role-specific.
- Profile data is role/name/traits only. Top-level `communication_style` and
  `notes` are rejected; future style/notes can live inside `traits`.
- Profile traits match `GlobalState.AgentProfile` and accept only string,
  integer, float, boolean, list-of-strings, or null values.
- `AgentProposal` remains final structured output only.
- `VerbalOnlyProposal` rejects non-null actions.
- Prompt builders include:
  - role
  - profile traits when provided
  - emotion context when provided
  - role-specific observation
  - recent messages
  - the common partial-observation warning
  - active-agent silence option
  - final structured JSON only instruction
  - no internal reasoning instruction
  - no chain-of-thought instruction
  - no `raw_text` instruction
  - instruction not to append internal reasoning or chain-of-thought to
    `runtime_state.messages`
- Role-specific prompt builders reject mismatched roles.
- Patient prompt adds patient-emotion context labeling, patient-internal-state
  guidance, free-form `disclosure_rules` guidance, hidden-fact disclosure
  caution, and communication-ability silence/answer guidance.
- Nurse prompt is verbal-only, says nurse has no physical action, and allows
  result-related reporting or bedside reassurance.
- Relative prompt is verbal-only, has no behavior action, says to stay silent
  unless asked or selected, and notes no relative emotion transition in
  v1.3.1.
- Clinician prompt allows one verbal action plus one action, restricts action
  choices to `medical_treatment_order`, `diagnostic_order`, or `null`, says
  `no_action` is not clinician-selectable, forbids `raw_text`, describes
  `family -> kind_hint -> params`, says diagnostic action selects one
  `test_name`, and allows null params.
- `parse_agent_proposal(...)` rejects:
  - `raw_text` anywhere
  - `internal_reasoning` anywhere
  - `chain_of_thought` anywhere
  - non-null actions for nurse, patient, and relative
- The parser preserves explicit `null` values and does not validate
  `kind_hint`, `test_name`, or params. Those checks remain `ActionValidator`
  responsibility.

## Deviations From `docs/v1_3_1.md`

- No intentional behavioral deviation.
- The v1.3.1 document describes proposal shapes but does not define dedicated
  Pydantic models. M8a adds the smallest shared models needed for future LLM
  agents.
- Generic parser coverage was included in M8a because it stayed small and
  boundary-focused. Full clinician parser/prompt integration remains deferred.
- `VerbalAction` continues to use the M6 `recipient` field. A separate
  `target` field was not reintroduced.

## Known Limitations

- No real LLM agents.
- No OpenAI or external API integration.
- No role-specific full agent classes.
- No turn-loop integration for prompt-built runtime inputs.
- No memory summarization or long-term memory.
- No behavior actions, task queues, async workflows, retry loops, or extra
  managers.
- The parser is generic and does not perform clinician action validation.
- Prompt helpers render text only; they do not guarantee clinical quality or
  role behavior.

## Exact Next Step

M8b Clinician LLM proposal parser/prompt integration.

## M8a_patch

- Tightened `ed_world_model/agents/schemas.py::AgentProfile.traits` to match
  `ed_world_model/state/global_state.py::AgentProfile.traits`:
  `dict[str, str | int | float | bool | list[str] | None]`.
- Added focused schema coverage for bounded trait values and rejection of
  complex nested trait objects.
- No code was added to store or return internal reasoning. Internal reasoning
  remains ephemeral future-agent implementation detail only; final
  `AgentProposal` output still rejects `internal_reasoning` and
  `chain_of_thought`.

Patch test runs:

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
21 passed in 0.06s
```

```bash
python -m pytest tests/test_turn_loop_stub.py tests/test_agent_schemas.py tests/test_agent_prompting.py
```

Result: passed.

```text
38 passed in 0.08s
```
