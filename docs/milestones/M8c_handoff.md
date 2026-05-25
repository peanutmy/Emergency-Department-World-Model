# M8c Handoff

## Files Created or Modified

- `ed_world_model/agents/patient.py`
  - Added `PatientAgent` with injected `llm_callable(prompt) -> str`.
  - Added `build_patient_prompt(...)`.
  - Added `parse_patient_response(...)` / `parse_patient_proposal(...)`.
  - Added `PatientParserError`.
- `ed_world_model/agents/nurse.py`
  - Added `NurseAgent` with injected `llm_callable(prompt) -> str`.
  - Added `build_nurse_prompt(...)`.
  - Added `parse_nurse_response(...)` / `parse_nurse_proposal(...)`.
  - Added `NurseParserError`.
- `ed_world_model/agents/relative.py`
  - Added `RelativeAgent` with injected `llm_callable(prompt) -> str`.
  - Added `build_relative_prompt(...)`.
  - Added `parse_relative_response(...)` / `parse_relative_proposal(...)`.
  - Added `RelativeParserError`.
- `ed_world_model/agents/prompting.py`
  - Added shared strict verbal-only response parsing for patient, nurse, and
    relative agents.
  - Normalizes LLM-facing `verbal_action.target` to the existing shared
    `VerbalAction.recipient` schema field.
  - Returns the existing `VerbalOnlyProposal` model.
- `tests/test_patient_agent.py`
  - Added focused patient prompt/parser/fake-callable tests and shared
    verbal-only architecture boundary tests.
- `tests/test_nurse_agent.py`
  - Added focused nurse prompt/parser/fake-callable tests.
- `tests/test_relative_agent.py`
  - Added focused relative prompt/parser/fake-callable tests.
- `docs/milestones/M8c_handoff.md`
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
- clinician agent behavior beyond shared parser compatibility support
- `StateManager`
- `Orchestrator`
- `ObservationBuilder`
- engine adapters

No real LLM API client, retry/repair loop, behavior action, nurse physical
action, relative emotion transition, scenario-loader change, or turn-loop
integration was added.

## Tests Run and Results

Focused M8c run:

```bash
python -m pytest tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py
```

Result: passed.

```text
46 passed in 0.14s
```

Adjacent agent/schema run:

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_clinician_agent.py tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py
```

Result: passed.

```text
91 passed in 0.13s
```

Manual import check:

```bash
python -c "import ed_world_model; import ed_world_model.agents.patient; import ed_world_model.agents.nurse; import ed_world_model.agents.relative"
```

Result: passed.

Broad pytest was not run because M8c only adds patient/nurse/relative
prompt/parser layers and the requested focused plus adjacent agent suites
passed.

## Patient/Nurse/Relative Prompt and Parser Behavior

- All three prompt builders use only:
  - role-specific partial observation,
  - optional `AgentProfile`,
  - recent messages,
  - optional patient emotion context for the patient agent.
- All three prompts tell the agent to silently reason before speaking about:
  - whether a verbal contribution is useful,
  - who it should target,
  - what content is appropriate from the partial observation,
  - how `AgentProfile` traits should shape the spoken content.
- Prompted reasoning is private generation context only. The prompts forbid
  outputting reasoning, rationale, analysis, `internal_reasoning`, or
  chain-of-thought.
- Patient prompt covers patient-owned hidden fields:
  `chief_complaint`, `symptoms`, `hidden_history`, `hidden_allergies`,
  `hidden_home_medications`, and `disclosure_rules`.
- Patient prompt states `hidden_*` fields are hidden from the clinical team,
  not from the patient, and that `disclosure_rules` are free-form guidance.
- Nurse prompt states the nurse is verbal-only, has no physical action, does
  not execute physical actions directly, and may report newly available
  results or bedside reassurance/instruction when present in observation.
- Relative prompt states the relative is verbal-only, does not perform
  behavior actions or block care, must not invent hidden clinical facts, must
  not read exact monitor-level vitals unless explicitly observed, and has no
  relative emotion transition in v1.3.1.
- The shared parser:
  - accepts strict JSON only,
  - requires the top-level `verbal_action` field,
  - allows `verbal_action: null`,
  - rejects duplicate JSON keys and JSON constants such as `NaN`,
  - rejects `raw_text`, `internal_reasoning`, and `chain_of_thought` anywhere,
  - rejects `action`, `medical_treatment_order`, `diagnostic_order`, and
    `behavior_action` anywhere,
  - rejects extra top-level fields,
  - normalizes `target` to the shared `recipient` field,
  - verifies `speaker` matches the parser role,
  - does not validate medical correctness,
  - does not call `ActionValidator`,
  - does not access or mutate `GlobalState`,
  - does not call engines.

## Deviations From `docs/v1_3_1.md` and Why

- No intentional architecture deviation was introduced.
- The requested LLM-facing response shape uses `verbal_action.target`, while
  the existing shared M6/M8a `VerbalAction` schema uses `recipient`. M8c
  accepts `target` at the verbal-only LLM boundary and normalizes it to
  `recipient` before returning the shared `VerbalOnlyProposal`, avoiding a
  duplicate incompatible proposal model.
- The parser is stricter than the generic M8a parser for verbal-only LLM output:
  it rejects any `action` field, including `action: null`, because the M8c
  response shape contains only `verbal_action`.

## Known Limitations

- No real LLM client is implemented.
- No turn-loop integration exists for the clinician/patient/nurse/relative
  LLM-style agents yet.
- No retry, repair, markdown extraction, memory summarization, or long-term
  memory exists.
- Prompt quality does not guarantee clinically correct or role-perfect speech.
- The patient prompt does not implement deterministic disclosure-rule logic;
  `disclosure_rules` remain free-form generation guidance.
- The parsers do not validate clinical truthfulness beyond structural role and
  forbidden-field checks.

## Exact Next Step

Integrate LLM-style clinician/patient/nurse/relative agents into the turn loop
with fake callables, or build a real-case demo runner.

## M8c_patch Prompt-Level Anti-Repetition

### Files Modified

- `ed_world_model/agents/prompting.py`
  - Added `ANTI_REPETITION_RULE` for shared prompt construction.
  - The generic shared prompt builder now includes this rule.
- `ed_world_model/agents/clinician.py`
  - Added the same prompt-level anti-repetition rule to the clinician prompt.
- `ed_world_model/agents/patient.py`
  - Added the same prompt-level anti-repetition rule to the patient prompt.
- `ed_world_model/agents/nurse.py`
  - Added the same prompt-level anti-repetition rule to the nurse prompt.
- `ed_world_model/agents/relative.py`
  - Added the same prompt-level anti-repetition rule to the relative prompt.
- `tests/test_agent_prompting.py`
  - Added shared prompt anti-repetition assertions.
- `tests/test_clinician_agent.py`
  - Added clinician prompt anti-repetition assertions.
- `tests/test_patient_agent.py`
  - Added patient prompt anti-repetition assertions.
- `tests/test_nurse_agent.py`
  - Added nurse prompt anti-repetition assertions.
- `tests/test_relative_agent.py`
  - Added relative prompt anti-repetition assertions.

### Behavior

- All four agent prompts now tell the agent to silently compare planned
  `verbal_action.content` with recent messages, especially its own prior
  messages.
- The prompts discourage repeating the same information, reassurance, question,
  concern, or instruction in different words.
- The prompts tell agents to return `verbal_action: null` when they have
  nothing new or useful to add.
- The prompts explicitly allow repetition when:
  - the agent is directly asked again,
  - the agent is correcting a misunderstanding,
  - the agent is confirming critical information,
  - new clinical or conversation context makes repetition necessary.
- This patch is prompt-level only. It does not add deterministic duplicate
  suppression, semantic similarity checks, embeddings, LLM-based judgment,
  retries, repair loops, new memory stores, or turn-loop changes.

### Tests Run and Results

```bash
python -m pytest tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py tests/test_agent_prompting.py tests/test_clinician_agent.py
```

Result: passed.

```text
84 passed in 0.20s
```

### Exact Next Step

Inspect trajectories using the prompt-level anti-repetition rule, then decide
whether a deterministic same-speaker duplicate guard is needed.

## M8c_patch Turn-Loop Compatibility Hardening

### Files Modified

- `ed_world_model/agents/patient.py`
  - `PatientAgent.generate(...)` now returns `AgentProposal`.
  - The parser still returns the internal verbal-only shape.
- `ed_world_model/agents/nurse.py`
  - `NurseAgent.generate(...)` now returns `AgentProposal`.
  - The parser still returns the internal verbal-only shape.
- `ed_world_model/agents/relative.py`
  - `RelativeAgent.generate(...)` now returns `AgentProposal`.
  - The parser still returns the internal verbal-only shape.
- `tests/test_patient_agent.py`
  - Added public generate-path `AgentProposal` assertions.
  - Added prompt assertions that the prompt does not include full
    `GlobalState`/`truth_state`/`runtime_state`.
  - Added parser hardening coverage for speaker mismatch, invalid target,
    simultaneous `target` and `recipient`, extra top-level fields, extra
    `verbal_action` fields, duplicate JSON keys, and JSON constants.
- `tests/test_nurse_agent.py`
  - Added matching public generate-path and parser hardening coverage.
- `tests/test_relative_agent.py`
  - Added matching public generate-path and parser hardening coverage.

### Behavior

- Public `PatientAgent.generate(...)`, `NurseAgent.generate(...)`, and
  `RelativeAgent.generate(...)` now return the existing shared
  `AgentProposal`, so they are compatible with the M6 turn-loop agent
  interface.
- Public generated proposals preserve verbal-only guarantees:
  - `action` is always `None`,
  - LLM output containing `action` is rejected,
  - `medical_treatment_order`, `diagnostic_order`, and `behavior_action` are
    rejected anywhere,
  - `raw_text`, `internal_reasoning`, and `chain_of_thought` are rejected
    anywhere.
- Parser internals still use `VerbalOnlyProposal` before converting the public
  generate output to `AgentProposal`.
- No turn-loop, clinician-agent, `ActionValidator`, `Orchestrator`,
  `ObservationBuilder`, `StateManager`, Hybrid adapter, engine, scenario, PDF,
  or output files were modified for this hardening patch.

### Helper Consolidation

- Broad helper consolidation was deferred. The role modules still have small
  local prompt formatting/profile helper functions.
- Only the minimal public-output conversion helper was added locally in each
  verbal-only role module.

### Tests Run and Results

Focused M8c patch run:

```bash
python -m pytest tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py
```

Result: passed.

```text
74 passed in 0.14s
```

Adjacent agent/schema run:

```bash
python -m pytest tests/test_agent_schemas.py tests/test_agent_prompting.py tests/test_clinician_agent.py tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py
```

Result: passed.

```text
119 passed in 0.13s
```

### Exact Next Step

Integrate LLM-style clinician/patient/nurse/relative agents into the turn loop
with fake callables, or run a full trajectory/demo to inspect prompt behavior.
