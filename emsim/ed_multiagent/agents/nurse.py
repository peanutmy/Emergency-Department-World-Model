"""Nurse agent skeleton for M5."""
from __future__ import annotations

from .base import BaseAgent


class NurseAgent(BaseAgent):
    """No-LLM nurse skeleton with explicit observation boundaries."""

    role = "nurse"
    role_instructions = (
        "You are an ED nurse.",
        "Report observable vitals from the current nurse observation only.",
        "Execute validated tasks only when assigned through deterministic workflow.",
        "Ask for clarification when orders or tasks are incomplete or ambiguous.",
        "Warn concisely when observable deterioration or pending events require it.",
    )
    invariant_role_constraints = (
        "Do not reveal hidden diagnosis or hidden physiology.",
        "Do not invent vitals; if a vital is absent from observation, say it is unavailable.",
        "Do not execute tasks outside the validated task queue.",
        "Do not assume you are called every simulation round.",
    )


__all__ = ["NurseAgent"]
