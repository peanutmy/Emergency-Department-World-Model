"""Simple v1.3.1 termination check."""
from __future__ import annotations

from ed_world_model.state.global_state import GlobalState


def is_terminated(global_state: GlobalState) -> bool:
    return (
        global_state.runtime_state.turn_index >= global_state.runtime_state.max_turns
        or global_state.patient_state.status_flags.is_alive is False
    )


def termination_reason(global_state: GlobalState) -> str | None:
    if global_state.runtime_state.turn_index >= global_state.runtime_state.max_turns:
        return "max_turns"
    if global_state.patient_state.status_flags.is_alive is False:
        return "patient_not_alive"
    return None


__all__ = ["is_terminated", "termination_reason"]
