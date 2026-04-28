# rule_engine Instructions

This directory is EMSim core physiology.

Prefer adapters under `ed_multiagent/emsim_adapter/` when building the multi-agent layer. The adapter should call or wrap EMSim; it should not move multi-agent workflow concerns into `rule_engine/`.

Do not change physiology formulas, drug pharmacodynamics, intervention effects, pathology drift, mapping behavior, or hidden-state semantics unless explicitly requested.

Do not add agent workflow, memory, prompt, LLM, or observation-gateway code here.

If a task explicitly requires a core EMSim change, keep it minimal, preserve existing transition behavior unless the task says otherwise, and add focused regression tests.

`rule_engine` may expose stable adapter-facing APIs, but EMSim remains the physiology source of truth.
