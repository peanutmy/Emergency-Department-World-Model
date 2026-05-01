# M9 Evaluation Rubric Context

## Goal

Define scenario-backed evaluation rubrics and scoring that assess clinical decisions, timing, workflow, communication, and unsafe actions from structured logs.

## Relevant V4 Design Sections

- Section 11: scenario format includes evaluation fields.
- Section 12: evaluation logger and metrics.
- Section 13: AI clinician evaluation path.
- Section 20.3: validator protects integrity while evaluator judges safety.
- Section 20.4 and 23.5: disposition, termination, and final outcome.
- Section 24.1.3: ACLS decision timing as an evaluation signal.

## Current Dependencies

- `ed_multiagent/evaluation/logger.py` records structured round entries.
- `ed_multiagent/orchestrator/simulation.py` produces orders, tasks, EMSim actions, observations, and memory deltas.
- `ed_multiagent/world/` tracks order/task timing and lab events.
- `ed_multiagent/memory/discovered.py` stores discovered clinical facts and test results.
- `ed_multiagent/actions/response_opportunity.py` tracks answered, partial, refused, evaded, ignored, and unable responses.
- Scenario/rubric loader code may not exist yet; add only when the implementation task requires it.

## Allowed Files/Folders

- `ed_multiagent/evaluation/`
- `ed_multiagent/scenario/`
- `tests/ed_multiagent/test_evaluation_rubric.py`
- `docs/` for rubric examples if requested

## Forbidden Scope

- Do not implement model-based grading or LLM evaluation unless a future task explicitly requests it.
- Do not alter runtime behavior to improve scores.
- Do not collapse structured clinical facts into transcript summaries.
- Do not make the evaluator depend on `rule_engine` internals or hidden physiology fields unavailable through logs/events.
- Do not block unsafe but executable actions; score them after logging.

## Required Behavior

- Rubrics should be data-driven per scenario: critical actions, unsafe actions, key history, reassessment expectations, disposition targets, and timing windows.
- Scoring should consume structured run logs, orders, tasks, physiology events, memory deltas, response opportunity statuses, and final outcome.
- Compute basic metrics for time-to-order, time-to-execution, lab turnaround, missing history, unsafe actions, reassessment, communication, and disposition.
- Warnings/rejections from validation should be reportable without changing whether executable actions proceeded.
- Rubric output should be deterministic and replayable from saved logs.

## Required Tests

- Rubric loads from scenario/evaluation data and validates required fields.
- Time-to-critical-order and time-to-execution metrics are computed from structured logs.
- Unsafe executable action is scored even if the validator allowed it.
- Missing allergy/history question and failed reassessment are detected from structured memory/opportunity data.
- Disposition/final outcome scoring works without reading hidden engine state.

## Acceptance Criteria

- A completed run can produce deterministic scores without LLM calls.
- Evaluation distinguishes structural invalid actions, clinically unsafe executable actions, workflow delays, and communication gaps.
- Rubric/scorer remains useful for both human-clinician demos and AI-clinician evaluation.

## Future Hybrid-Engine Compatibility Constraints

- The current EMSim rule-based engine may later be replaced by a hybrid engine.
- Evaluation should consume engine-agnostic logs, `EngineAdvanceResult` physiology events, observations, workflow events, and structured memory.
- Do not score by importing `rule_engine` objects or inspecting backend hidden state directly.
- Hybrid-engine-specific metrics must be added through explicit logged events or normalized scenario outcomes.
