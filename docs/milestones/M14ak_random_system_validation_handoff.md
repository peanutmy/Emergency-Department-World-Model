# M14ak Random System Validation Handoff

## Request and preparation

User requested five randomized whole-system runs, especially emotion, and asked
which Claude or GPT API credentials are available. Read AGENTS.md and the M14aj
and M14ai handoffs. No runtime, engine, scenario, or existing test files changed.

Selected five scenarios with seed 20261027; saved the selection in
`validation_runs/20261027_random5/manifest.json`. Plan six turns per scenario,
subject to normal early termination. Live runs have NOT been executed.

## Credential blocker

Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is present in the process or Windows
User/Machine environment. No root .env file was found. Project VS Code and Claude
settings do not reference these API keys. Secret values were not displayed.
This does not establish whether the user owns keys elsewhere.
Asked for provider choice and the existing credential configuration location.

The existing `generate_utterances.py` entry point supports `--emotion`,
`--provider`, `--scenario`, `--turns`, `--trait-seed`, `--output`, and
`--physiology-mode hybrid`. Its emotion path injects the selected provider into
both patient and relative emotion adapters and the other model-backed components.
Use this entry point after credentials are configured; do not substitute fake
runs for the requested live whole-system validation.

## Offline checks

Ran:

```text
python -B -m pytest -p no:cacheprovider tests/test_full_turn_loop_integration.py tests/test_emotion_rules_consistency.py tests/test_noop_emotion.py -q
```

Result: 42 passed, 1 failed. Failure:
`test_step_prompt_embeds_shared_definitions_verbatim` checks that definitions
are literal in the raw template, which now contains a formatting placeholder.
The previous M14aj handoff already documented this obsolete assertion. Left
existing tests unchanged. Import check passed. These tests do not establish
live emotion scoring quality or successful live physiology transitions.

## Credential setup follow-up

User saved OpenAI settings in Windows User environment. An elevated read now
confirms the key exists (not printed), model is `gpt-5.5`, and base URL is
`https://148.113.224.153:3000/v1`. The sandbox cannot see these user settings;
load them into Process scope outside the sandbox for authorized live execution.
Installed openai and anthropic SDKs into the existing `.venvWMER` environment.

HTTPS connection failed during TLS handshake with `SSL: WRONG_VERSION_NUMBER`.
An unauthenticated HTTP probe to the same host/port returned 401, confirming an
HTTP service. No API key was sent over HTTP. Live runs remain unexecuted pending
a working HTTPS URL or explicit user approval to send credentials over HTTP.
