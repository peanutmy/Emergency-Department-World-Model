# M7 Patient And Relative LLM Context

## Goal

Add LLM-backed patient and relative behavior grounded in subjective state, private memory, emotion, and open response opportunities.

## Relevant V4 Design Sections

- Sections 7 and 7.4: `ResponseOpportunity`, response modes, and non-answer semantics.
- Section 8 and 20.6: emotion state and two-stage affect updates.
- Sections 9.4 and 9.5: patient and relative prompt principles.
- Section 20.2 and 23.6: `SubjectiveState` versus `SymptomTimeline`.
- Sections 23.3.1, 23.3.2, and 24.2: CODE-mode patient/relative activation policies.

## Current Dependencies

- `ed_multiagent/agents/patient.py` and `relative.py` contain no-LLM skeletons.
- `ed_multiagent/observation/gateway.py` provides patient and relative observations.
- `ed_multiagent/observation/subjective.py` defines `SubjectiveState` and `SymptomTimeline`.
- `ed_multiagent/memory/private_memory.py` defines patient and relative private memory.
- `ed_multiagent/actions/response_opportunity.py` owns response status and structured memory updates.
- `ed_multiagent/orchestrator/activation_policy.py` skips/throttles patient and relative activation in CODE mode.

## Allowed Files/Folders

- `ed_multiagent/agents/patient.py`
- `ed_multiagent/agents/relative.py`
- `ed_multiagent/agents/llm_client.py`
- `ed_multiagent/affect/`
- `ed_multiagent/actions/response_opportunity.py` only for structured extraction/status integration
- `tests/ed_multiagent/test_patient_relative_agents_llm.py`

## Forbidden Scope

- Do not let patient or relative see hidden diagnosis, exact vitals, hidden physiology, raw pathology, or rule-engine internals unless staff communicated them through structured memory.
- Do not let patient or relative mutate EMSim physiology directly.
- Do not add AI clinician behavior.
- Do not use LLM summaries or transcripts as the clinical fact source of truth.
- Do not bypass speech-capacity limits from `SubjectiveState`.

## Required Behavior

- Patient speaks from current `SubjectiveState`, `SymptomTimeline`, patient private memory, emotion, recent dialogue, and current response opportunity.
- Relative speaks from visible patient state, relative private memory, emotion, recent dialogue, and current response opportunity.
- Patient speech capacity constrains response mode; `unable` must force unable-to-answer behavior.
- Direct and partial answers update `DiscoveredClinicalMemory` through structured slots, not transcript parsing.
- Refusal, evasion, silence, emotional reaction, and unable-to-answer must not update clinical facts.
- CODE mode skips unresponsive patient LLM calls and throttles relative LLM calls unless a major event or open opportunity requires activation.

## Required Tests

- Mocked patient LLM respects `speech_capacity == "unable"`.
- Patient answer can update structured symptom/history slots only when direct or partial.
- Relative can provide collateral history that exists in relative private memory.
- Refusal/evasion/silence does not update discovered clinical facts.
- CODE-mode activation skip/throttle behavior is respected.

## Acceptance Criteria

- Patient and relative LLMs enrich dialogue without breaking information boundaries.
- Structured memory remains the source of clinical facts.
- Tests run with mocked LLM responses and no network dependency.

## Future Hybrid-Engine Compatibility Constraints

- The current EMSim rule-based engine may later be replaced by a hybrid engine.
- Patient and relative agents must consume only `ObservationGateway` outputs, `SubjectiveState`, response opportunities, and structured private/discovered memory.
- Do not import from `rule_engine`, inspect hidden state, or depend on current EMSim hidden field names.
- Any future hybrid engine must satisfy the same observation and subjective-state contracts before agents consume it.
