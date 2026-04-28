# EMSim Repository Instructions

EMSim is the physiology source of truth. Vitals, rhythms, drug effects, intervention effects, and hidden physiology must come from EMSim or deterministic adapters around EMSim.

The multi-agent layer must not invent vitals or hidden state. Agents may speak, request, order, report, refuse, or react, but they do not directly author physiology.

Clinical facts must be structured. Allergies, medications, symptom timing, vitals history, test results, orders, and events belong in structured memory or logs, not only in conversation summaries.

Unsafe but executable clinical actions should be allowed and logged, not blocked by the validator. The validator protects system integrity; the evaluator judges clinical safety; EMSim simulates consequences.

Do not implement LLM calls unless the current task explicitly requests LLM integration. Before that point, use deterministic mocks, templates, fixtures, or scripted agents.

Do not modify EMSim core physiology unless explicitly requested. Prefer adding multi-agent adapters under `ed_multiagent/`.

Add tests for new behavior. New workflow, memory, observation, validation, mode, orchestration, and agent contracts should have focused tests.

Implement only the requested milestone. Do not skip ahead to later agents, UI, scoring, prompt caching, or autonomous clinician behavior unless the task asks for it.

Follow the v4 design in `docs/emsim_multi_agent_system_plan_v4.md` and the task breakdown in `docs/codex_implementation_tasks.md`.
