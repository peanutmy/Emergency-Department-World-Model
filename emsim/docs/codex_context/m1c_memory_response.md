# M1c Memory And Response Context

## Goal

Add structured memory and response opportunities with hardcoded no-LLM role behavior.

## Relevant Sections From V4

- Section 4: Memory system and permission boundaries.
- Section 6: structured action system.
- Section 7: `ResponseOpportunity` and response modes.
- Section 20.5: clinical facts must be structured, not only summaries.
- Section 23.8: M1c no-LLM full role loop.

## Allowed Files/Folders

- `ed_multiagent/memory/`
- `ed_multiagent/actions/`
- `ed_multiagent/agents/mock.py`
- `tests/ed_multiagent/`

## Forbidden Scope

- Do not implement real LLM agents.
- Do not expose ground truth, hidden diagnosis, hidden physiology, or undiscovered allergy to clinician/nurse.
- Do not use conversation summaries as the source of clinical facts.
- Do not mutate EMSim state from memory or response logic.

## Required Tests

- Clinician cannot see undiscovered allergy.
- Clinician question creates a response opportunity.
- Direct or partial answer updates `DiscoveredClinicalMemory`.
- Refusal, evasion, silence, and unable-to-answer do not update clinical facts.

## Acceptance Criteria

- Information asymmetry works without LLMs.
- Clinical facts are stored in structured memory.
- Hardcoded patient/relative/nurse scripts can drive response opportunities.
