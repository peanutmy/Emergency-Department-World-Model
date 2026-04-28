"""
EMSim Physiology Engine — interface + baseline stub.

Replace `IdentityEngine` below with your real rule-based implementation.
The evaluation harness (emsim_eval.py) only depends on the `.step()` signature.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol


# --- Type aliases (states/actions are plain dicts matching schema.json) ---
State = dict[str, Any]
Action = dict[str, Any]


class PhysiologyEngine(Protocol):
    """Minimum interface any engine must implement to be evaluated."""

    def step(
        self,
        before: State,
        actions: list[Action],
        duration_s: float,
    ) -> State:
        """
        Predict the `after` state given the `before` state, a list of concurrent
        actions taken at t=0, and an observation window of `duration_s` seconds.

        Must return a complete state dict: {vitals, interventions, drugs, mechanism}.
        """
        ...


class IdentityEngine:
    """
    Baseline: returns `before` unchanged (ignores actions and time).

    Purpose: sanity-check the eval harness plumbing and provide a floor.
    Expect pass-rate ≈ rate of pairs where nothing changes (pure-wait pairs
    with no drift, or no-op branches). Everything above this floor comes from
    your rules.
    """

    def step(
        self,
        before: State,
        actions: list[Action],
        duration_s: float,
    ) -> State:
        return deepcopy(before)
