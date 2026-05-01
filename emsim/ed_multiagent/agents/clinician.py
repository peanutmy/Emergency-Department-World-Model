"""Clinician agent skeleton for M5."""
from __future__ import annotations

from .base import BaseAgent


class ClinicianAgent(BaseAgent):
    """No-LLM clinician skeleton for future structured decision output."""

    role = "clinician"
    role_instructions = (
        "You are the clinician.",
        "You only know discovered clinical information.",
        "You cannot access hidden diagnosis or hidden physiology.",
        "Output structured orders, questions, and information requests.",
        "Specify drug, dose, route, and priority when an order is eventually supported.",
    )
    invariant_role_constraints = (
        "Do not infer undiscovered facts from the physiology engine.",
        "Do not invent vital signs, test results, or hidden diagnosis.",
        "Physical execution must remain in deterministic workflow systems.",
        "Do not assume you are called every simulation round.",
    )


__all__ = ["ClinicianAgent"]
