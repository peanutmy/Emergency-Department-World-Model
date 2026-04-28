"""
rule_engine — rule-based physiology engine for EMSim.

Public exports:
    RuleEngine     — L1 core transition API (emsim_eval.py entry point)
    EngineSession  — L2 agent-facing stateful wrapper (Phase 0.5)
"""
from .engine import RuleEngine
from .session import EngineSession

__all__ = ["RuleEngine", "EngineSession"]
