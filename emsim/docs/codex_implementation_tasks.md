# Codex Implementation Tasks

Source of truth: `docs/emsim_multi_agent_system_plan_v4.md`.

This task list is sequential. Do not skip ahead, do not implement LLM calls before the explicit LLM tasks, and do not modify EMSim physiology unless a task explicitly requires it.

## Task 0: Repository Analysis Only, No Code Changes

### Goal

Understand the current EMSim repository structure, existing rule engine interfaces, transition schema, tests, and v4 design requirements before implementation.

### Files Likely To Create/Modify

None.

### Required Behavior

- Read `docs/emsim_multi_agent_system_plan_v4.md`.
- Inspect `rule_engine/`, `transitions/schema.json`, existing tests, and any packaging/test configuration.
- Produce an implementation map that identifies reusable EMSim interfaces and missing APIs.

### Forbidden Scope

- Do not create, modify, format, or delete files.
- Do not implement `ed_multiagent/`.
- Do not alter EMSim rule engine behavior.

### Tests Required

- None, because this is analysis-only.

### Acceptance Criteria

- Codex reports the existing reusable interfaces.
- Codex identifies where `EngineSession` should wrap/call EMSim.
- Codex identifies missing multi-agent APIs and implementation risks.

## Task 1: Project Instructions And Context Files

### Goal

Create repository-level implementation guidance and milestone-specific context files so future Codex tasks can run with bounded scope.

### Files Likely To Create/Modify

- `AGENTS.md`
- `rule_engine/AGENTS.md`
- `docs/codex_implementation_tasks.md`
- `docs/codex_context/*.md`

### Required Behavior

- Document milestone order, allowed scope, forbidden scope, tests, and acceptance criteria.
- Explicitly state that EMSim is the physiology source of truth.
- Explicitly state that unsafe but executable clinical actions should be allowed and logged, not blocked by the validator.

### Forbidden Scope

- Do not create `ed_multiagent/` code.
- Do not implement Python modules.
- Do not modify EMSim behavior.
- Do not add LLM calls.

### Tests Required

- Documentation review only.
- Verify the requested files exist and contain the required sections.

### Acceptance Criteria

- All requested docs and AGENTS files exist.
- Every context file is short, task-specific, and usable as standalone task context.
- No runtime code is created or modified.

## Task 2: M1a EngineSession Wrapper

### Goal

Wrap EMSim as a live `EngineSession` that persists patient physiology over simulated time.

### Files Likely To Create/Modify

- `ed_multiagent/emsim_adapter/engine_session.py`
- `ed_multiagent/emsim_adapter/__init__.py`
- `tests/ed_multiagent/test_engine_session.py`
- Possibly `rule_engine/session.py`, only if explicitly approved for EMSim core integration.

### Required Behavior

- Construct an `EngineSession` from an initial EMSim schema state.
- Expose observable vitals without exposing hidden physiology to agents.
- Apply EMSim-compatible actions in the existing schema format.
- Advance physiology with `advance(dt_s)` and preserve state across calls.
- Return an `EngineAdvanceResult`-style object with current state and physiology events once the event contract is introduced.

### Forbidden Scope

- Do not add workflow, orders, nurse tasks, memory, observations, or agents.
- Do not change physiology formulas, drug PD, intervention effects, pathology drift, or hidden-state semantics.
- Do not invent vitals outside EMSim.

### Tests Required

- Session construction from a valid initial state.
- Repeated `advance(dt_s)` calls persist state and time.
- `apply_NRB`, `give_fluids`, and at least one drug action can be applied.
- Agents or observations receive vitals only, not hidden state.

### Acceptance Criteria

- M1a can load a scenario initial state, run 30 physiology-only rounds, and record vitals drift.
- No workflow or multi-agent code is required to use the session.
- Existing EMSim transition behavior remains unchanged.

## Task 3: M1b WorkflowEngine / Orders / Nurse Task Queue

### Goal

Add deterministic clinical workflow plumbing: hand-typed orders become nurse tasks, completed tasks become EMSim actions, and physiology changes after execution delay.

### Files Likely To Create/Modify

- `ed_multiagent/world/state.py`
- `ed_multiagent/world/orders.py`
- `ed_multiagent/world/nurse_tasks.py`
- `ed_multiagent/world/events.py`
- `ed_multiagent/world/workflow_engine.py`
- `ed_multiagent/emsim_adapter/action_mapper.py`
- `tests/ed_multiagent/test_workflow_engine.py`
- `tests/ed_multiagent/test_order_to_emsim_action.py`

### Required Behavior

- Represent `WorldState`, `ClinicalOrder`, `NurseTask`, and pending events.
- Keep clinician physical actions as orders, not immediate EMSim mutations.
- Use `NurseTask.remaining_s` measured in simulated seconds.
- Enforce MVP nurse capacity of one active task unless explicitly changed.
- Convert completed drug/intervention tasks into EMSim-compatible actions.
- Model MVP lab turnaround as scheduled events.

### Forbidden Scope

- Do not add LLM agents.
- Do not add role-specific observation gateway beyond what tests need.
- Do not alter EMSim core physiology.
- Do not implement full medication safety validation beyond structural mapping.

### Tests Required

- Hand-typed drug order creates a nurse task and executes after delay.
- Intervention order maps to a valid EMSim action.
- Lab order schedules and returns a result after configured turnaround.
- Task timing decrements by simulated seconds, not rounds.

### Acceptance Criteria

- A scripted order can move through order manager, nurse task queue, EMSim action mapper, and `EngineSession`.
- Vitals change only after the nurse task completes and EMSim advances.
- Logs or events can show order creation, task creation, and task completion.

## Task 4: M1c Memory + ResponseOpportunity Without LLM

### Goal

Add structured clinical memory, conversation memory, response opportunities, and hardcoded role responses without any LLM calls.

### Files Likely To Create/Modify

- `ed_multiagent/memory/ground_truth.py`
- `ed_multiagent/memory/discovered.py`
- `ed_multiagent/memory/conversation.py`
- `ed_multiagent/memory/private_memory.py`
- `ed_multiagent/memory/event_memory.py`
- `ed_multiagent/actions/schema.py`
- `ed_multiagent/actions/response_opportunity.py`
- `tests/ed_multiagent/test_memory_response.py`

### Required Behavior

- Store true clinical facts separately from discovered clinical facts.
- Store critical clinical facts structurally, not only in free-text summaries.
- Create `ResponseOpportunity` objects from questions.
- Support direct answer, partial answer, refusal, evasion, silence, and unable-to-answer states.
- Update `DiscoveredClinicalMemory` only for direct or partial answers.
- Use hardcoded scripts/templates for patient, relative, and nurse responses.

### Forbidden Scope

- Do not implement LLM agents.
- Do not expose ground truth, hidden diagnosis, or hidden physiology to clinician/nurse memory.
- Do not use conversation summaries as clinical source of truth.
- Do not implement emotion beyond simple fields needed for response tests.

### Tests Required

- Clinician cannot see undiscovered allergy.
- Question creates a response opportunity.
- Hardcoded patient or relative answer updates discovered memory.
- Refusal/evasion/silence does not update discovered clinical facts.

### Acceptance Criteria

- Information asymmetry is enforced through structured memory.
- A no-LLM role loop can ask allergy/history questions and update discovered facts correctly.
- Clinical facts remain queryable without parsing a transcript.

## Task 5: M1d SimModeManager + CODE Activation Policy

### Goal

Add managed simulation modes, dynamic time steps, CODE-mode activation policy, human slash commands, and ROSC event contract scaffolding without LLM calls.

### Files Likely To Create/Modify

- `ed_multiagent/orchestrator/sim_mode.py`
- `ed_multiagent/orchestrator/activation_policy.py`
- `ed_multiagent/orchestrator/acls.py`
- `ed_multiagent/actions/parser.py`
- `ed_multiagent/actions/schema.py`
- `ed_multiagent/world/workflow_engine.py`
- `tests/ed_multiagent/test_sim_mode.py`
- `tests/ed_multiagent/test_activation_policy.py`
- `tests/ed_multiagent/test_code_timing.py`

### Required Behavior

- Implement `SimMode.STABLE`, `SimMode.URGENT`, and `SimMode.CODE`.
- Upgrade modes immediately when thresholds require it.
- Downgrade from urgent only after hysteresis.
- Exit CODE only with ROSC plus explicit `exit_code` or scenario-specific criteria.
- Use `get_round_dt_s`: stable 30s, urgent 10s, code 5s.
- Centralize `should_call_clinician_agent`, `should_call_nurse_agent`, `should_call_patient_agent`, and `should_call_relative_agent`.
- Distinguish deterministic workflow execution from optional LLM/verbal activation.
- Parse `/code` and `/exit_code` into structured actions.

### Forbidden Scope

- Do not add LLM calls.
- Do not implement full ACLS clinical reasoning.
- Do not change EMSim physiology to force ROSC or arrest.
- Do not block unsafe but executable actions.

### Tests Required

- Frozen hysteresis fixture from v4.
- Stable to urgent upgrade happens immediately.
- Urgent does not oscillate near HR threshold.
- Arrest rhythm or `/code` triggers CODE.
- `/exit_code` without ROSC does not downgrade CODE.
- Mock ROSC plus `/exit_code` downgrades CODE to urgent.
- Patient activation is skipped when unresponsive in CODE.
- Relative activation throttles to every 30s unless major event.
- Clinician activation is event-driven in CODE, not every 5s.

### Acceptance Criteria

- Dynamic time step is controlled by stateful mode manager.
- CODE mode advances workflow/physiology at high frequency while cognition/dialogue remains event-triggered.
- Task timing remains stable across mode transitions.

## Task 6: SubjectiveState Renderer

### Goal

Convert EMSim physiology into current patient subjective experience for patient behavior and speech capacity.

### Files Likely To Create/Modify

- `ed_multiagent/emsim_adapter/subjective.py`
- `ed_multiagent/memory/private_memory.py`
- `tests/ed_multiagent/test_subjective_renderer.py`

### Required Behavior

- Define `SubjectiveState` with dyspnea, confusion, palpitation, dizziness, pain distress, speech capacity, and visible distress.
- Render subjective state from EMSim hidden state and observable vitals through the adapter layer.
- Treat speech capacity as a physical constraint on response mode.
- Keep longitudinal symptom history in private memory or `SymptomTimeline`, not in `SubjectiveState`.

### Forbidden Scope

- Do not expose raw hidden variables to agents.
- Do not let patient agent choose speech capacity freely when physiology says unable.
- Do not modify EMSim core unless explicitly requested.
- Do not add LLM calls.

### Tests Required

- Severe hypoxia or low consciousness reduces speech capacity.
- SVT/VT or high chronotropic drive yields palpitations.
- Low BP or low perfusion yields dizziness.
- Patient observation contains subjective state, not hidden fields.

### Acceptance Criteria

- Patient-facing state is derived deterministically from EMSim physiology.
- The renderer can be used by response opportunity logic and observation gateway.

## Task 7: ObservationGateway

### Goal

Build role-specific observations from world, EMSim, memory, workflow, response opportunities, and subjective state.

### Files Likely To Create/Modify

- `ed_multiagent/observation/gateway.py`
- `ed_multiagent/observation/role_views.py`
- `ed_multiagent/observation/symptom_renderer.py`
- `tests/ed_multiagent/test_observation_gateway.py`

### Required Behavior

- Clinician sees discovered facts, recent vitals history, orders, results, and pending clarifications.
- Nurse sees current vitals, pending orders, current task, queue, and recent conversation.
- Patient sees symptoms, subjective state, emotion, recent conversation, and relevant response opportunity.
- Relative sees visible patient state, relative private knowledge, emotion, conversation, and relevant response opportunity.
- Future ECG/imaging results should fit without rewriting the gateway.

### Forbidden Scope

- Do not expose hidden state, hidden diagnosis, undiscovered allergy, drug PD internals, or raw pathology label to clinician/nurse.
- Do not expose exact numeric vitals to patient unless communicated.
- Do not add LLM calls.
- Do not mutate state from observation builders.

### Tests Required

- Clinician cannot see undiscovered ground-truth allergy or hidden diagnosis.
- Patient observation omits numeric vitals by default.
- Nurse observation includes current vitals and task queue.
- Relative observation contains relative private knowledge but no hidden physiology.

### Acceptance Criteria

- Each role receives only permitted information.
- Observation generation is deterministic and side-effect free.

## Task 8: OrderValidator

### Goal

Validate orders for structural executability while allowing unsafe but executable clinical decisions to proceed and be logged/evaluated.

### Files Likely To Create/Modify

- `ed_multiagent/actions/validator.py`
- `ed_multiagent/actions/schema.py`
- `ed_multiagent/memory/event_memory.py`
- `ed_multiagent/evaluation/scorer.py`
- `tests/ed_multiagent/test_order_validator.py`

### Required Behavior

- Return statuses `ALLOW`, `WARN`, or `REJECT`.
- Reject structurally impossible orders such as unknown drug, invalid route, missing required dose, or unknown intervention action.
- Warn but allow executable high-risk orders, known discovered allergies, unusual doses, or guideline deviations.
- Do not warn for allergies that exist only in undiscovered ground truth.
- Log warnings and rejected actions for evaluation.

### Forbidden Scope

- Do not block clinically unsafe but executable orders.
- Do not implement a full medication safety database.
- Do not mutate EMSim state directly.
- Do not add LLM calls.

### Tests Required

- Unknown drug is rejected.
- Missing dose rejects or requires clarification.
- Known discovered allergy returns warn and remains executable.
- Undiscovered allergy does not warn.
- WPW contraindication-style unsafe order is allowed and logged for evaluator.

### Acceptance Criteria

- Validator protects system integrity without hiding consequences of unsafe clinical choices.
- Evaluator can score unsafe actions separately from validator decisions.

## Task 9: Scenario Loader

### Goal

Introduce interactive scenario format and loader separate from transition-pair calibration data.

### Files Likely To Create/Modify

- `ed_multiagent/scenario/schema.py`
- `ed_multiagent/scenario/loader.py`
- `ed_multiagent/scenario/scenarios/*.yaml`
- `tests/ed_multiagent/test_scenario_loader.py`

### Required Behavior

- Load scenario metadata, patient/relative private memory, ground truth, initial EMSim state, tests, and evaluation rubric.
- Validate required EMSim initial state fields against the existing transition schema shape.
- Provide at least one WPW, one sepsis, and one respiratory/anaphylaxis-style scenario when implementation scope allows.

### Forbidden Scope

- Do not treat transition pairs as full interactive scripts.
- Do not add LLM calls.
- Do not implement orchestration beyond loader tests.
- Do not modify `transitions/schema.json` unless explicitly requested.

### Tests Required

- Valid scenario loads into a typed spec.
- Invalid missing initial vitals fails validation.
- Ground truth remains separate from discovered memory.
- Test turnaround values are loaded.

### Acceptance Criteria

- M1a and later tasks can construct a simulation from a scenario spec.
- Scenario format supports patient, relative, ground truth, initial state, tests, and evaluation fields.

## Task 10: No-LLM Orchestrator Loop

### Goal

Connect scenario, engine session, workflow, memory, observation, response opportunities, SimMode, and logging into a deterministic no-LLM round loop.

### Files Likely To Create/Modify

- `ed_multiagent/orchestrator/round_loop.py`
- `ed_multiagent/orchestrator/simulation.py`
- `ed_multiagent/main.py`
- `tests/ed_multiagent/test_no_llm_orchestrator.py`

### Required Behavior

- Parse human clinician input into structured actions.
- Add verbal actions to conversation memory.
- Create response opportunities from questions.
- Validate and enqueue orders.
- Advance nurse tasks, execute completed EMSim actions, and advance physiology.
- Fire pending events and lab results.
- Build observations and use hardcoded/mock agents.
- Update discovered memory and emotion state.
- Advance world time using the active SimMode `dt_s`.

### Forbidden Scope

- Do not add real LLM calls.
- Do not create autonomous four-agent free-for-all behavior.
- Do not bypass order/task delays.
- Do not let non-EMSim components invent vitals.

### Tests Required

- Hypoxia case: order `apply_NRB`, nurse executes after delay, O2 changes through EMSim, log records sequence.
- Allergy question: response opportunity updates discovered memory.
- Lab order returns after turnaround and updates discovered memory.
- CODE mode uses dynamic time step and activation policy.

### Acceptance Criteria

- A deterministic no-LLM case can run end to end with replayable state changes.
- The round loop follows clinician turn plus environment reaction, not simultaneous free agents.

## Task 11: Evaluation Logger

### Goal

Make simulation runs replayable and evaluable with structured logs and basic scoring.

### Files Likely To Create/Modify

- `ed_multiagent/evaluation/logger.py`
- `ed_multiagent/evaluation/replay.py`
- `ed_multiagent/evaluation/scorer.py`
- `tests/ed_multiagent/test_evaluation_logger.py`

### Required Behavior

- Record round id, time, clinician input, parsed actions, orders, tasks, EMSim actions, vitals before/after, response opportunities, agent turns, discovered-memory delta, emotion delta, and rejected/warned actions.
- Emit JSON run logs.
- Support replay or deterministic inspection of a trajectory.
- Compute basic workflow, clinical decision, communication, and unsafe-action metrics.

### Forbidden Scope

- Do not implement model-based grading.
- Do not add LLM evaluation.
- Do not change runtime behavior just to improve scoring.
- Do not collapse structured clinical facts into summaries.

### Tests Required

- Round log contains all required top-level fields.
- Unsafe warning is recorded without blocking execution.
- Replay can reconstruct key timeline events.
- Time-to-order and time-to-execution metrics are computed.

### Acceptance Criteria

- Every no-LLM orchestrator run produces a structured replayable log.
- Evaluation can inspect unsafe actions, workflow delays, and decision timing.

## Task 12: BaseAgent Static/Dynamic Prompt Contract, Mock Agents Only

### Goal

Define the agent interface and prompt boundary before any real LLM integration.

### Files Likely To Create/Modify

- `ed_multiagent/agents/base.py`
- `ed_multiagent/agents/mock.py`
- `ed_multiagent/agents/nurse.py`
- `ed_multiagent/agents/patient.py`
- `ed_multiagent/agents/relative.py`
- `ed_multiagent/agents/clinician.py`
- `tests/ed_multiagent/test_base_agent_contract.py`

### Required Behavior

- Define `BaseAgent.build_static_prompt()`.
- Define `BaseAgent.build_dynamic_prompt(observation)`.
- Define `BaseAgent.act(observation)` returning structured `AgentTurnOutput`.
- Provide mock/template agents only.
- Ensure prompts receive structured observations and response opportunities.

### Forbidden Scope

- Do not call any LLM provider.
- Do not implement provider clients, API keys, streaming, retries, or prompt caching infrastructure.
- Do not add autonomous clinician behavior beyond mocks.
- Do not expose hidden state to prompts.

### Tests Required

- Every mock agent implements static and dynamic prompt methods.
- `act()` returns structured output.
- Dynamic prompt changes with observation while static prompt remains invariant.
- Hidden state and undiscovered facts are absent from prompt inputs.

### Acceptance Criteria

- The LLM integration boundary exists and is testable without network calls.
- Future LLM agents can plug into the same structured action contract.

## Task 13: NurseAgent LLM Integration

### Goal

Integrate the nurse as the first real LLM agent while keeping task execution deterministic.

### Files Likely To Create/Modify

- `ed_multiagent/agents/nurse.py`
- `ed_multiagent/agents/llm_client.py`
- `ed_multiagent/config.py`
- `tests/ed_multiagent/test_nurse_agent_llm.py`

### Required Behavior

- Nurse LLM can report vitals, ask clarification, warn on deterioration, and explain current task status.
- Nurse task execution remains controlled by `NurseTaskQueue`, not by LLM text.
- Nurse output must parse to structured `AgentTurnOutput`.
- In CODE mode, nurse verbalization is called only when activation policy allows it.

### Forbidden Scope

- Do not let NurseAgent execute tasks outside the workflow engine.
- Do not let NurseAgent invent vitals.
- Do not add patient, relative, or clinician LLMs.
- Do not modify EMSim physiology.

### Tests Required

- Mocked LLM response parses to structured nurse action.
- Invalid LLM output is rejected or repaired without mutating state.
- Nurse cannot report hidden diagnosis or hidden physiology.
- CODE activation policy can skip nurse LLM while task execution continues.

### Acceptance Criteria

- Nurse LLM affects communication only.
- Workflow execution remains deterministic and testable.

## Task 14: PatientAgent And RelativeAgent LLM Integration

### Goal

Add patient and relative LLM behavior grounded in subjective state, private memory, emotion, and response opportunities.

### Files Likely To Create/Modify

- `ed_multiagent/agents/patient.py`
- `ed_multiagent/agents/relative.py`
- `ed_multiagent/affect/emotion_state.py`
- `ed_multiagent/affect/affect_engine.py`
- `tests/ed_multiagent/test_patient_relative_agents_llm.py`

### Required Behavior

- Patient speaks from subjective state, symptom timeline, private memory, emotion, and current response opportunity.
- Relative speaks from relative private memory, visible state, emotion, and current response opportunity.
- Patient speech capacity constrains possible response modes.
- Direct/partial answers update discovered memory through structured extraction, not transcript summaries.
- CODE mode skips patient if unresponsive and throttles relative unless major event.

### Forbidden Scope

- Do not let patient or relative know hidden diagnosis or exact vitals unless told.
- Do not let patient or relative mutate EMSim physiology directly.
- Do not add AI clinician.
- Do not use LLM summaries as clinical fact source of truth.

### Tests Required

- Mocked patient LLM respects unable-to-answer speech capacity.
- Relative can provide known collateral history.
- Refusal/evasion does not update discovered clinical facts.
- CODE activation policy skips/throttles as expected.

### Acceptance Criteria

- Patient and relative LLMs enrich dialogue without breaking information boundaries.
- Structured memory remains the source of clinical facts.

## Task 15: AI ClinicianAgent Integration

### Goal

Add AI clinician as the final LLM agent after deterministic simulation, workflow, memory, observation, validation, logging, and other agents are stable.

### Files Likely To Create/Modify

- `ed_multiagent/agents/clinician.py`
- `ed_multiagent/actions/parser.py`
- `ed_multiagent/evaluation/scorer.py`
- `tests/ed_multiagent/test_clinician_agent_llm.py`

### Required Behavior

- AI clinician receives only discovered clinical information and allowed observations.
- AI clinician outputs structured verbal actions, information actions, and order bundles.
- AI clinician can use structured code declarations and slash-command-equivalent actions.
- In CODE mode, clinician activation is ACLS decision-event driven, not every 5s.
- Evaluator scores unsafe actions, timing, missing history, reassessment, and disposition.

### Forbidden Scope

- Do not let AI clinician access ground truth, hidden physiology, or hidden diagnosis.
- Do not accept free text as the only clinical action output.
- Do not block unsafe but executable actions in validator.
- Do not change EMSim core physiology to make agent behavior look better.

### Tests Required

- Mocked AI clinician output parses to structured action bundle.
- Hidden state and undiscovered facts are absent from clinician prompt.
- Unsafe but executable order is allowed, logged, and scored.
- CODE decision-event cadence prevents clinician calls every 5s.

### Acceptance Criteria

- AI clinician can run a scenario through the existing deterministic simulator.
- Logs are sufficient to evaluate clinical quality and safety.
- The system remains EMSim-backed, structured, replayable, and role-permissioned.
