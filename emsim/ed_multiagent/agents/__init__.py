"""Agent contracts and deterministic mock agents for EMSim multi-agent M5."""

from .base import BaseAgent
from .clinician import ClinicianAgent
from .mock import MockAgent
from .nurse import NurseAgent
from .patient import PatientAgent
from .relative import RelativeAgent

__all__ = [
    "BaseAgent",
    "ClinicianAgent",
    "MockAgent",
    "NurseAgent",
    "PatientAgent",
    "RelativeAgent",
]
