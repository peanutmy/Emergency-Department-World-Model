"""
Engine event dataclasses (ENGINE_DESIGN.md §6.5 L2 API).

v1: defined but unused. EngineSession will emit them in Phase 0.5, and
the orchestrator will route them once it exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Something state-changing the engine did (not caused by an action).

    `kind` values in use: "arrest", "rosc", "seizure", "arrhythmia_change",
    "vital_alarm", "compensation_failure".
    """
    kind: str
    t_sim_s: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionReceipt:
    """Returned from `EngineSession.submit_action` (ENGINE_DESIGN.md §6.5)."""
    success: bool
    message: str = ""
    immediate_vital_change: dict[str, float] = field(default_factory=dict)
