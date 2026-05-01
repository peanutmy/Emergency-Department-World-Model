"""Relative agent skeleton for M5."""
from __future__ import annotations

from .base import BaseAgent


class RelativeAgent(BaseAgent):
    """No-LLM relative skeleton with collateral-history boundaries."""

    role = "relative"
    role_instructions = (
        "You are the patient's relative.",
        "You know only relative private memory and visible patient state.",
        "You may provide collateral history, ask questions, or react emotionally.",
        "You do not know hidden diagnosis or physiology.",
    )
    invariant_role_constraints = (
        "Do not invent exact vitals or hidden clinical facts.",
        "Use only visible patient state, recent dialogue, response opportunities, and relative private memory.",
        "Do not assume you are called every simulation round.",
    )


__all__ = ["RelativeAgent"]
