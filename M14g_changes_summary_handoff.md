# M14g — Changes Summary & Handoff

Concise summary of all changes made in this work session: emotion system, agent
traits, orchestration, clinician/patient prompts, a runnable end-to-end utterance
generator, and scenario data completion.

---

## Design baseline established
- 4 agents: **patient, relative, clinician, nurse**.
- Only **patient and relative** are emotional agents (have emotion + traits).
  Clinician and nurse have **no emotion, no traits** (they only drive the sim).
- Emotion model: 4 dims `fear / sadness / overwhelm / shame`, 4 levels
  `not at all / low / high / extreme`. Patient and relative share the same schema.
- Emotion lives in `GlobalState.psych_state`, flows to agents **only via the
  observation** (no separate `emotion_context` channel). Traits live in
  `agent_profiles`.

---

## 1. Random agent traits at scenario load
- **NEW** `ed_world_model/agents/agent_traits.py` — `AgentTraitLibrary`: loads
  per-role trait JSON, samples 1..N random traits + a random level each.
- `ed_world_model/scenario_loader.py` — assigns random traits at load time, only
  to **patient/relative** (`TRAIT_BEARING_ROLES`); `trait_seed` for reproducibility.

## 2. Emotion state model (patient + relative)
- `ed_world_model/state/global_state.py` — removed the broken `PatientEmotion`
  (6 duplicate fields collapsed to 1); added `AgentEmotion` (4 dims + `notes` +
  `summary()`) and `EmotionLevel`; `PsychState` now holds `patient_emotion` **and**
  `relative_emotion`.
- `ed_world_model/state/state_manager.py` — `update_patient_emotion` rewritten to
  the new schema; added `update_relative_emotion`.
- `ed_world_model/adapters/emotion_engine_adapter.py` — fixed broken imports
  (root `emotion_engine`, dropped non-existent `EmotionEntry`); `AgentEmotion ↔
  EmotionState` direct field copy.
- `ed_world_model/orchestration/runner.py` — `IntegrationRunner` wires a
  `relative_emotion_engine`.
- `ed_world_model/orchestration/observation_builder.py` — relative observation
  carries `relative_emotion`.

## 3. Unify emotion on observation (removed `emotion_context`)
- Removed the dead `emotion_context` parameter end-to-end:
  `ed_world_model/agents/schemas.py`, `prompting.py`, `patient.py`,
  `clinician.py`, `relative.py`. Emotion now reaches prompts only through the
  observation; relative prompt now uses emotion for tone.

## 4. Traits only for patient/relative
- **DELETED** `ed_world_model/agents/agentTrait/ClinicianTrait.json` and
  `NurseTrait.json`. `agent_traits.py::ROLE_TRAIT_FILES` reduced to patient/relative.

## 5. Orchestration — make patient/relative speak when addressed
- `ed_world_model/orchestration/orchestrator.py` — new `RECIPIENT_LAST_TURN`
  rule: the patient is activated whenever it was the recipient of any message in
  the previous turn (not only on `requires_response`).

## 6. Clinician prompt fixes (anti-flatline)
- `ed_world_model/agents/clinician.py` `build_clinician_decision_prompt`:
  (1) do not go idle while the patient is unstable; (2) escalate when current
  treatment is failing; (3) prefer addressing the patient directly; (4) reference
  vitals naturally/rounded instead of telemetry dumps.

## 7. Emotion init = "not at all" (no LLM seed)
- `ed_world_model/adapters/emotion_engine_adapter.py` — new
  `seed_initial_emotion=False` (default). Initial emotion is all `not at all`;
  no `init_emotion` LLM call; emotions move only via `step_emotion`.

## 8. Emotion accounts for consciousness (freeze when sedated/intubated)
- `ed_world_model/orchestration/turn_loop.py`:
  - `_suppresses_consciousness(action)` → when a **sedation / RSI / intubation /
    surgical-airway** treatment runs, set `is_conscious=False`, `can_speak=False`,
    `features.intubated=True`.
  - Patient emotion is **frozen** (engine not called) while the patient is dead or
    unconscious. The relative (a separate person) keeps updating.

## 9. Relative emotion only when the relative is present
- `ed_world_model/orchestration/turn_loop.py` — relative emotion is updated only
  if the relative has actually spoken (is in the scene); otherwise untracked.
- `generate_utterances.py` — output/print omits `relative_emotion` /
  `relative_traits` entirely when the relative never appears.

## 10. End-to-end runner + standalone script
- **NEW** `generate_utterances.py` — runs a scenario through the real-agent demo
  and prints/saves utterances + emotion. Supports OpenAI/Anthropic relays,
  per-model param adaptation (`gpt-5*`/`o*` reasoning models), 429 retry/backoff,
  `--emotion` (real emotion engine), `--physiology-mode hybrid|fake`,
  `--trait-seed`, `--output`.
- **NEW** `examples/run_emotion_trajectory.py` — standalone emotion-engine
  trajectory tester (scripted dialogue, fake or real LLM).

## 11. Scenario data completion (from PDFs)
- **NEW** `fill_scenarios.py` (one-off) — filled `chief_complaint`, `symptoms`
  (lowercase, so it loads), `baseline_vitals`, and per-finding `turnaround_turns`
  (by `test_category`) for **one scenario per transitions subfolder** (9 cases),
  grounded in each PDF.
- Fixed `transitions/Cardiology/Acute Respiratory Distress.json`: `Symptoms` →
  `symptoms` (capital key never loaded).

---

## Tests & docs
- Updated tests: `test_global_state`, `test_noop_emotion`, `test_state_manager`,
  `test_observation_builder`, `test_turn_loop_stub`, `test_patient_agent`,
  `test_clinician_agent`, `test_agent_prompting`, `test_relative_agent`,
  `test_orchestrator`. Suite: **740 passed** (3 pre-existing unrelated failures:
  2 `test_action_registry`, 1 `test_loader_does_not_import_transition_engines`).
- Docs: `docs/emotion_trait_refactor_changelog.md` (with per-file diffs),
  `docs/agent_utterance_frequency.md`.

## How to run the full sim
```bash
# env in ~/.zshrc: OPENAI_API_KEY, OPENAI_BASE_URL (relay /v1)
AGENT_LLM_MODEL=gpt-4o python generate_utterances.py \
  --provider openai --emotion --turns 15 --trait-seed 1 \
  --physiology-mode hybrid --output run.json
```

## Latest run result (run_full_real_retry.json, hybrid, 15 turns)
Working: no flatline, real escalation with hybrid physiology response
(O2 80→95, RR 34→12 after intubation), consciousness freeze, relative appears +
emotion gated correctly, clinician asks the relative for history once the patient
is intubated, init = not at all.

## Known open issues (next candidates)
1. Once the patient can't answer, the clinician asks the relative repeatedly and
   the relative replies "I don't know" for many turns → late-run repetition loop.
   Needs: stop questioning after ~2 "don't know"s; wind down once the patient is
   stable (let the clinician go idle).
2. Frozen patient emotion sits at `high` while sedated — consider zeroing to
   `not at all` when unconscious instead of holding the last value.
3. `sadness` never activates with no-LLM init (under-reactive); `fear` jitters
   low↔high turn to turn (sticky/step calibration).
4. Patient still speaks relatively little even with recipient activation: the
   Stage-1 `should_speak` bias is the remaining gate (a "answer when directly
   asked" bias was recently added to `patient.py`).
