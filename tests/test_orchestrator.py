from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.orchestration.orchestrator as orchestrator_module
from ed_world_model.orchestration.orchestrator import (
    DEFAULT_CLINICIAN_ACTIVE,
    EXPLICITLY_SELECTED,
    NEWLY_AVAILABLE_RESULTS,
    REQUIRED_RESPONSE,
    Orchestrator,
)
from ed_world_model.state.global_state import DiagnosticResult, Event, GlobalState


def test_clinician_is_active_by_default_on_empty_state() -> None:
    decision = Orchestrator().select_active_agents(GlobalState())

    assert decision.active_agents == ["clinician"]
    assert decision.activation_reasons == {
        "clinician": [DEFAULT_CLINICIAN_ACTIVE],
    }
    assert decision.required_response_agents == []


def test_required_response_agents_are_activated() -> None:
    state = GlobalState(
        runtime_state={"required_response_agents": ["patient", "nurse"]}
    )

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician", "nurse", "patient"]
    assert decision.required_response_agents == ["nurse", "patient"]
    assert decision.activation_reasons["nurse"] == [REQUIRED_RESPONSE]
    assert decision.activation_reasons["patient"] == [REQUIRED_RESPONSE]


def test_patient_is_activated_when_required_to_respond() -> None:
    state = GlobalState(runtime_state={"required_response_agents": ["patient"]})

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician", "patient"]
    assert decision.activation_reasons["patient"] == [REQUIRED_RESPONSE]


def test_nurse_is_activated_when_required_to_respond() -> None:
    state = GlobalState(runtime_state={"required_response_agents": ["nurse"]})

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician", "nurse"]
    assert decision.activation_reasons["nurse"] == [REQUIRED_RESPONSE]


def test_nurse_is_activated_when_newly_available_results_is_non_empty() -> None:
    state = GlobalState(
        runtime_state={
            "newly_available_results": [
                DiagnosticResult(name="ECG", result="atrial fibrillation")
            ]
        }
    )

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician", "nurse"]
    assert decision.activation_reasons["nurse"] == [NEWLY_AVAILABLE_RESULTS]


def test_relative_is_not_active_by_default() -> None:
    decision = Orchestrator().select_active_agents(GlobalState())

    assert "relative" not in decision.active_agents
    assert "relative" not in decision.activation_reasons


def test_relative_is_active_when_required_to_respond() -> None:
    state = GlobalState(runtime_state={"required_response_agents": ["relative"]})

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician", "relative"]
    assert decision.activation_reasons["relative"] == [REQUIRED_RESPONSE]


def test_relative_can_be_active_when_explicitly_selected() -> None:
    decision = Orchestrator().select_active_agents(
        GlobalState(),
        explicitly_selected_agents={"relative"},
    )

    assert decision.active_agents == ["clinician", "relative"]
    assert decision.activation_reasons["relative"] == [EXPLICITLY_SELECTED]


def test_active_agents_has_no_duplicates() -> None:
    state = GlobalState(
        runtime_state={
            "required_response_agents": ["clinician", "nurse", "nurse"],
            "newly_available_results": [
                DiagnosticResult(name="ECG", result="atrial fibrillation")
            ],
        }
    )

    decision = Orchestrator().select_active_agents(
        state,
        explicitly_selected_agents={"relative"},
    )

    assert decision.active_agents == ["clinician", "nurse", "relative"]
    assert len(decision.active_agents) == len(set(decision.active_agents))


def test_activation_reasons_explain_why_each_agent_was_activated() -> None:
    state = GlobalState(
        runtime_state={
            "required_response_agents": ["clinician", "nurse", "patient"],
            "newly_available_results": [
                DiagnosticResult(name="Troponin", result="elevated")
            ],
        }
    )

    decision = Orchestrator().select_active_agents(
        state,
        explicitly_selected_agents={"relative"},
    )

    assert decision.activation_reasons == {
        "clinician": [DEFAULT_CLINICIAN_ACTIVE, REQUIRED_RESPONSE],
        "nurse": [REQUIRED_RESPONSE, NEWLY_AVAILABLE_RESULTS],
        "patient": [REQUIRED_RESPONSE],
        "relative": [EXPLICITLY_SELECTED],
    }


def test_nurse_bedside_verbal_slot_is_not_selected_by_orchestrator() -> None:
    state = GlobalState(
        runtime_state={
            "current_turn_events": [
                Event(
                    type="nurse_shadow_execution",
                    payload={"execution_mode": "shadow_execution"},
                )
            ],
            "last_turn_events": [
                Event(
                    type="nurse_bedside_verbal_slot",
                    payload={"selected": True},
                )
            ],
        }
    )

    decision = Orchestrator().select_active_agents(state)

    assert decision.active_agents == ["clinician"]
    assert "nurse" not in decision.activation_reasons
    assert "nurse_bedside_verbal_slot" not in decision.active_agents


def test_orchestrator_does_not_mutate_global_state() -> None:
    state = GlobalState(
        runtime_state={
            "required_response_agents": ["patient"],
            "newly_available_results": [
                DiagnosticResult(name="ECG", result="atrial fibrillation")
            ],
        }
    )
    before = deepcopy(state.model_dump())

    Orchestrator().select_active_agents(
        state,
        explicitly_selected_agents={"relative"},
    )

    assert state.model_dump() == before


def test_orchestrator_does_not_import_transition_engines() -> None:
    source = inspect.getsource(orchestrator_module)

    assert "transition_engines" not in source


def test_orchestrator_does_not_call_state_manager() -> None:
    source = inspect.getsource(orchestrator_module)

    assert "StateManager" not in source
    assert "state_manager" not in source


def test_orchestrator_does_not_build_observations() -> None:
    source = inspect.getsource(orchestrator_module)

    assert "ObservationBuilder" not in source
    assert "build_for" not in source
