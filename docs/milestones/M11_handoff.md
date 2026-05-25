# M11 Handoff

## Files Created or Modified

- `ed_world_model/agents/llm_client.py`
  - Added `LLMClient` protocol with `generate(prompt: str) -> str`.
  - Added `FakeLLMClient` with deterministic scripted responses, prompt
    recording, `generate(...)`, callable behavior, and `complete(...)`
    compatibility for existing transition engines.
  - Added `OpenAICompatibleLLMClient` / `RealLLMClient` alias.
  - Real client reads `OPENAI_API_KEY` from the environment, raises a clear
    missing-key error, exposes `generate(...)` and `complete(...)`, and redacts
    the API key from request exception messages.
- `ed_world_model/demo.py`
  - Added public runtime switches:
    - `--agent-mode fake|real`
    - `--physiology-mode fake|hybrid`
  - Kept fake agents and fake physiology as defaults.
  - Kept hidden backward-compatible `--mode fake` support.
  - Updated default fake demo agents to use `FakeLLMClient`.
  - Added real-agent construction using `OpenAICompatibleLLMClient` behind
    explicit real mode.
  - Added Hybrid physiology construction using the existing
    `HybridPhysiologyAdapter` behind explicit hybrid mode.
  - Hybrid adapter import is lazy so fake default mode does not instantiate the
    transition-engine stack.
- `tests/test_llm_client.py`
  - Added focused client wrapper tests for fake scripting, missing API key,
    mocked real-client generation, and API-key redaction.
- `tests/test_demo_runner.py`
  - Added coverage for default fake/fake modes, real-agent mode with mocked
    client construction, missing-key failure, Hybrid physiology mode with a
    stub adapter, parser-error drop/log behavior, and CLI switch parsing.
- `tests/test_full_turn_loop_integration.py`
  - Added coverage that LLM-client-backed clinician output still flows through
    the existing clinician parser and validator.
- `docs/milestones/M11_handoff.md`
  - Added this handoff.

No files under `transition_engines/` or `transitions/` were modified.

## How to Run Fake Mode

```bash
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5
```

Equivalent explicit form:

```bash
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5 \
  --agent-mode fake \
  --physiology-mode fake
```

## How to Run Real Agent Mode

```bash
export OPENAI_API_KEY=...
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5 \
  --agent-mode real \
  --physiology-mode fake
```

Real agent output still goes through the existing clinician, patient, nurse,
and relative parsers. Parser failures are logged as `validation_drop` events
and the agent is treated as silent for that turn.

## How to Run Hybrid Physiology Mode

```bash
export OPENAI_API_KEY=...
python examples/run_demo_scenario.py \
  --scenario tests/fixtures/scenario_loader_minimal.json \
  --turns 5 \
  --agent-mode fake \
  --physiology-mode hybrid
```

Hybrid mode uses the existing `HybridPhysiologyAdapter`. Runtime actions still
carry `raw_text: None`, including system-generated `no_action`.

## Required Environment Variables

- `OPENAI_API_KEY`: required only when `--agent-mode real` or
  `--physiology-mode hybrid` needs the real OpenAI-compatible client.

No API key is required for default fake mode or for the test suite.

## Tests Run and Results

```bash
python -m pytest tests/test_llm_client.py tests/test_demo_runner.py tests/test_full_turn_loop_integration.py
```

Result: passed.

```text
42 passed in 0.51s
```

```bash
python -m pytest tests/test_clinician_agent.py tests/test_patient_agent.py tests/test_nurse_agent.py tests/test_relative_agent.py tests/test_hybrid_physiology_adapter.py
```

Result: passed.

```text
107 passed in 0.12s
```

CLI smoke:

```bash
python examples/run_demo_scenario.py --scenario tests/fixtures/scenario_loader_minimal.json --turns 1 --agent-mode fake --physiology-mode fake
```

Result: passed and printed a one-turn readable trajectory.

Verification:

```bash
git diff -- transition_engines transitions
```

Result: no diff.

## Test API-Safety Notes

- Tests do not require `OPENAI_API_KEY`.
- Tests do not call real provider APIs.
- Real-client tests inject mocked SDK clients or monkeypatch the demo real
  client class.
- Hybrid physiology demo-switch tests monkeypatch a stub
  `HybridPhysiologyAdapter`.
- Missing-key behavior is tested with the real client constructor and no API
  call.

## Known Limitations

- No prompt repair, retry loop, JSON repair, semantic duplicate suppression,
  async execution, token accounting, or cost accounting was added.
- Real agent quality depends on prompt behavior and has not been tuned against
  real trajectories yet.
- The real OpenAI-compatible client uses a simple Responses API call and does
  not enforce a provider-side JSON schema for agent outputs.
- Hybrid physiology mode uses the existing Hybrid adapter and engine behavior;
  no transition-engine routing or transition-pair runtime trajectory logic was
  added.
- Fake physiology remains deterministic demo behavior, not a clinical model.

## Exact Next Step

Real trajectory run and prompt tuning.
