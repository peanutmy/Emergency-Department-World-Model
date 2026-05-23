from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.orchestration.turn_loop as turn_loop_module
from ed_world_model.actions.registry import ActionFamily, ActionRegistry, KindHint
from ed_world_model.actions.validator import ActionValidator
from ed_world_model.agents.stubs import (
    ScriptedClinicianAgent,
    ScriptedNurseAgent,
    ScriptedPatientAgent,
    ScriptedRelativeAgent,
    SilentAgent,
)
from ed_world_model.orchestration.observation_builder import ObservationBuilder
from ed_world_model.orchestration.turn_loop import TurnLoop
from ed_world_model.state.global_state import Event, GlobalState, PendingDiagnosticResult
from ed_world_model.state.state_manager import StateManager


class FakePhysiologyAdapter:
    def __init__(self, output: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.output = output

    def predict(self, global_state: GlobalState, engine_facing_action: dict) -> dict:
        self.calls.append(
            {
                "action": deepcopy(engine_facing_action),
                "turn_index": global_state.runtime_state.turn_index,
                "known_results": [
                    result.model_dump()
                    for result in global_state.known_facts.available_results
                ],
            }
        )
        if self.output is not None:
            return deepcopy(self.output)
        return {"vitals": global_state.patient_state.vitals.model_dump()}


class FakeEmotionEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(
        self,
        current_patient_emotion,
        conversation_input=None,
        recent_messages=None,
        patient_profile_context=None,
    ):
        self.calls.append(
            {
                "current_patient_emotion": current_patient_emotion.model_dump(),
                "conversation_input": deepcopy(conversation_input),
                "recent_messages": deepcopy(recent_messages),
                "patient_profile_context": deepcopy(patient_profile_context),
            }
        )
        return current_patient_emotion.model_copy(deep=True)


class RecordingObservationBuilder(ObservationBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def build_for(self, active_agents, global_state):
        active_agents = list(active_agents)
        self.calls.append(
            {
                "active_agents": active_agents,
                "newly_available_results": [
                    result.model_dump()
                    for result in global_state.runtime_state.newly_available_results
                ],
            }
        )
        return super().build_for(active_agents, global_state)


def _state_with_test_bank(**runtime_state) -> GlobalState:
    return GlobalState(
        truth_state={
            "test_bank": [
                {
                    "name": "ECG",
                    "result": "atrial fibrillation",
                    "turnaround_turns": 1,
                },
                {
                    "name": "Chest X-ray",
                    "result": "pulmonary edema",
                    "turnaround_turns": 2,
                },
            ]
        },
        runtime_state=runtime_state,
    )


def _loop(
    manager: StateManager,
    physiology_adapter: FakePhysiologyAdapter | None = None,
    emotion_engine: FakeEmotionEngine | None = None,
    agents: dict | None = None,
    observation_builder: ObservationBuilder | None = None,
) -> tuple[TurnLoop, FakePhysiologyAdapter, FakeEmotionEngine]:
    physiology_adapter = physiology_adapter or FakePhysiologyAdapter()
    emotion_engine = emotion_engine or FakeEmotionEngine()
    loop = TurnLoop(
        state_manager=manager,
        physiology_adapter=physiology_adapter,
        agents=agents or {},
        observation_builder=observation_builder or ObservationBuilder(),
        action_validator=ActionValidator(ActionRegistry()),
        emotion_engine=emotion_engine,
    )
    return loop, physiology_adapter, emotion_engine


def _oxygen_order() -> dict:
    return {
        "type": "medical_treatment_order",
        "family": ActionFamily.RESPIRATORY_SUPPORT,
        "kind_hint": KindHint.OXYGEN_SUPPORT,
        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
    }


def test_clinician_active_by_default_and_silent_turn_uses_no_action() -> None:
    clinician = SilentAgent()
    manager = StateManager(GlobalState())
    loop, physiology, emotion = _loop(
        manager,
        agents={"clinician": clinician},
    )

    result = loop.run_turn()

    assert result.completed is True
    assert result.active_agents == ["clinician"]
    assert len(clinician.generated_observations) == 1
    assert physiology.calls[-1]["action"] == {
        "raw_text": None,
        "kind_hint": "no_action",
        "params": {"elapsed_min": 1},
    }
    assert result.physiology_action_kind_hint == "no_action"
    assert emotion.calls == []


def test_diagnostic_order_creates_pending_and_uses_no_action_physiology() -> None:
    clinician = ScriptedClinicianAgent(
        [{"action": {"type": "diagnostic_order", "test_name": "ECG"}}]
    )
    manager = StateManager(_state_with_test_bank())
    loop, physiology, _ = _loop(manager, agents={"clinician": clinician})

    result = loop.run_turn()

    assert manager.state.runtime_state.pending_diagnostic_results == [
        PendingDiagnosticResult(test_name="ECG", ordered_at_turn=0, ready_at_turn=1)
    ]
    assert manager.state.known_facts.available_results == []
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"
    assert not any(
        event["type"] == "nurse_bedside_verbal_slot" for event in result.events
    )


def test_diagnostic_release_happens_before_orchestrator_selection() -> None:
    state = _state_with_test_bank(turn_index=1)
    state.runtime_state.pending_diagnostic_results = [
        PendingDiagnosticResult(test_name="ECG", ordered_at_turn=0, ready_at_turn=1)
    ]
    manager = StateManager(state)
    observation_builder = RecordingObservationBuilder()
    loop, physiology, _ = _loop(
        manager,
        agents={"clinician": SilentAgent(), "nurse": SilentAgent()},
        observation_builder=observation_builder,
    )

    result = loop.run_turn()

    assert result.active_agents == ["clinician", "nurse"]
    assert result.released_diagnostics == [
        {"name": "ECG", "result": "atrial fibrillation"}
    ]
    assert observation_builder.calls[-1]["newly_available_results"] == [
        {"name": "ECG", "result": "atrial fibrillation"}
    ]
    assert manager.state.known_facts.available_results[0].name == "ECG"
    assert manager.state.runtime_state.newly_available_results == []
    assert physiology.calls[-1]["known_results"] == [
        {"name": "ECG", "result": "atrial fibrillation"}
    ]


def test_medical_treatment_order_records_shadow_execution_and_treatment_action() -> None:
    clinician = ScriptedClinicianAgent([{"action": _oxygen_order()}])
    manager = StateManager(GlobalState())
    loop, physiology, _ = _loop(manager, agents={"clinician": clinician})

    result = loop.run_turn()

    shadow_events = [
        event for event in result.events if event["type"] == "nurse_shadow_execution"
    ]
    assert shadow_events
    assert shadow_events[-1]["payload"]["ordered_by"] == "clinician"
    assert shadow_events[-1]["payload"]["executed_by"] == "nurse"
    assert shadow_events[-1]["payload"]["execution_mode"] == "shadow_execution"
    assert physiology.calls[-1]["action"]["kind_hint"] == "oxygen_support"
    assert physiology.calls[-1]["action"]["raw_text"] is None


def test_treatment_partial_feature_update_preserves_existing_features() -> None:
    state = GlobalState(
        patient_state={
            "features": {
                "rhythm": "NSR",
                "oxygen_device": None,
                "FiO2": 0.21,
                "intubated": False,
                "vent": False,
                "PEEP_cmH2O": None,
                "neuro_status": "GCS 15",
                "glucose_mmol_l": 6.2,
            }
        }
    )
    clinician = ScriptedClinicianAgent([{"action": _oxygen_order()}])
    physiology = FakePhysiologyAdapter(
        output={
            "features": {
                "oxygen_device": "NRB",
                "FiO2": 0.8,
            }
        }
    )
    manager = StateManager(state)
    loop, _, _ = _loop(
        manager,
        physiology_adapter=physiology,
        agents={"clinician": clinician},
    )

    loop.run_turn()

    assert manager.state.patient_state.features.model_dump() == {
        "rhythm": "NSR",
        "oxygen_device": "NRB",
        "FiO2": 0.8,
        "intubated": False,
        "vent": False,
        "PEEP_cmH2O": None,
        "neuro_status": "GCS 15",
        "glucose_mmol_l": 6.2,
    }


def test_nurse_bedside_verbal_slot_after_valid_treatment() -> None:
    clinician = ScriptedClinicianAgent([{"action": _oxygen_order()}])
    nurse = ScriptedNurseAgent(
        [
            {
                "verbal_action": {
                    "speaker": "nurse",
                    "content": "I am placing the oxygen mask now.",
                }
            }
        ]
    )
    manager = StateManager(GlobalState())
    loop, _, _ = _loop(
        manager,
        agents={"clinician": clinician, "nurse": nurse},
    )

    result = loop.run_turn()

    assert len(nurse.generated_observations) == 1
    assert manager.state.runtime_state.messages[-1].speaker == "nurse"
    assert manager.state.runtime_state.messages[-1].recipient == "patient"
    assert manager.state.runtime_state.messages[-1].content == (
        "I am placing the oxygen mask now."
    )
    bedside_events = [
        event for event in result.events if event["type"] == "nurse_bedside_verbal_slot"
    ]
    assert bedside_events[-1]["payload"]["spoke"] is True


def test_bedside_slot_records_spoke_true_when_nurse_was_active_at_turn_start() -> None:
    state = _state_with_test_bank(turn_index=1)
    state.runtime_state.pending_diagnostic_results = [
        PendingDiagnosticResult(test_name="ECG", ordered_at_turn=0, ready_at_turn=1)
    ]
    clinician = ScriptedClinicianAgent([{"action": _oxygen_order()}])
    nurse = ScriptedNurseAgent(
        [
            {
                "verbal_action": {
                    "speaker": "nurse",
                    "recipient": "patient",
                    "content": "I am placing this oxygen mask now.",
                }
            }
        ]
    )
    manager = StateManager(state)
    loop, _, _ = _loop(
        manager,
        agents={"clinician": clinician, "nurse": nurse},
    )

    result = loop.run_turn()

    assert result.active_agents == ["clinician", "nurse"]
    assert len(nurse.generated_observations) == 1
    assert manager.state.runtime_state.messages[-1].speaker == "nurse"
    assert manager.state.runtime_state.messages[-1].content == (
        "I am placing this oxygen mask now."
    )
    bedside_events = [
        event for event in result.events if event["type"] == "nurse_bedside_verbal_slot"
    ]
    assert bedside_events[-1]["payload"]["spoke"] is True


def test_bedside_slot_records_spoke_false_when_nurse_is_silent() -> None:
    clinician = ScriptedClinicianAgent([{"action": _oxygen_order()}])
    nurse = ScriptedNurseAgent([None])
    manager = StateManager(GlobalState())
    loop, _, _ = _loop(
        manager,
        agents={"clinician": clinician, "nurse": nurse},
    )

    result = loop.run_turn()

    bedside_events = [
        event for event in result.events if event["type"] == "nurse_bedside_verbal_slot"
    ]
    assert bedside_events[-1]["payload"]["spoke"] is False
    assert manager.state.runtime_state.messages == []


def test_diagnostic_order_does_not_invoke_nurse_bedside_slot() -> None:
    clinician = ScriptedClinicianAgent(
        [{"action": {"type": "diagnostic_order", "test_name": "ECG"}}]
    )
    nurse = ScriptedNurseAgent(["bedside statement should not run"])
    manager = StateManager(_state_with_test_bank())
    loop, _, _ = _loop(
        manager,
        agents={"clinician": clinician, "nurse": nurse},
    )

    loop.run_turn()

    assert nurse.generated_observations == []
    assert manager.state.runtime_state.messages == []


def test_invalid_clinician_action_is_dropped_and_no_diagnostic_created() -> None:
    clinician = ScriptedClinicianAgent(
        [
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "recipient": "patient",
                    "content": "I will check that.",
                },
                "action": {"type": "diagnostic_order", "test_name": "Unknown"},
            }
        ]
    )
    manager = StateManager(_state_with_test_bank())
    loop, physiology, _ = _loop(manager, agents={"clinician": clinician})

    result = loop.run_turn()

    assert manager.state.runtime_state.pending_diagnostic_results == []
    assert manager.state.runtime_state.messages[-1].content == "I will check that."
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"
    assert any(event["type"] == "validation_drop" for event in result.events)


def test_verbal_actions_commit_without_internal_reasoning_storage() -> None:
    clinician = ScriptedClinicianAgent(
        [
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "recipient": "patient",
                    "content": "How is your breathing?",
                    "requires_response": True,
                }
            }
        ]
    )
    patient = ScriptedPatientAgent(
        [
            {
                "verbal_action": {
                    "speaker": "patient",
                    "content": "It is tight.",
                }
            }
        ]
    )
    manager = StateManager(
        GlobalState(runtime_state={"required_response_agents": ["patient"]})
    )
    loop, _, _ = _loop(
        manager,
        agents={"clinician": clinician, "patient": patient},
    )

    loop.run_turn()

    assert [message.speaker for message in manager.state.runtime_state.messages] == [
        "clinician",
        "patient",
    ]
    dumped_messages = [
        message.model_dump() for message in manager.state.runtime_state.messages
    ]
    assert "internal_reasoning" not in str(dumped_messages)
    assert "chain" not in str(dumped_messages)


def test_emotion_engine_called_only_when_verbal_messages_committed() -> None:
    speaking_manager = StateManager(GlobalState())
    speaking_clinician = ScriptedClinicianAgent(
        [
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "content": "We are helping you.",
                }
            }
        ]
    )
    speaking_loop, _, speaking_emotion = _loop(
        speaking_manager,
        agents={"clinician": speaking_clinician},
    )

    speaking_loop.run_turn()

    assert len(speaking_emotion.calls) == 1
    assert speaking_manager.state.psych_state.patient_emotion.label == (
        speaking_emotion.calls[0]["current_patient_emotion"]["label"]
    )

    silent_loop, _, silent_emotion = _loop(
        StateManager(GlobalState()),
        agents={"clinician": SilentAgent()},
    )
    silent_loop.run_turn()

    assert silent_emotion.calls == []


def test_advance_turn_behavior_after_completed_turn() -> None:
    state = _state_with_test_bank(turn_index=1)
    state.runtime_state.pending_diagnostic_results = [
        PendingDiagnosticResult(test_name="ECG", ordered_at_turn=0, ready_at_turn=1)
    ]
    manager = StateManager(state)
    clinician = ScriptedClinicianAgent(
        [
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "content": "The ECG is back.",
                }
            }
        ]
    )
    loop, _, _ = _loop(manager, agents={"clinician": clinician, "nurse": SilentAgent()})

    result = loop.run_turn()

    assert manager.state.runtime_state.turn_index == 2
    assert manager.state.runtime_state.newly_available_results == []
    assert manager.state.runtime_state.messages[-1].content == "The ECG is back."
    assert manager.state.known_facts.available_results[0].name == "ECG"
    assert manager.state.runtime_state.last_turn_events
    assert manager.state.runtime_state.last_turn_events == [
        Event.model_validate(event) for event in result.events
    ]


def test_observations_and_invocation_only_for_active_agents() -> None:
    patient = ScriptedPatientAgent(["I should not be called."])
    nurse = ScriptedNurseAgent(["I should not be called."])
    relative = ScriptedRelativeAgent(["I should not be called."])
    observation_builder = RecordingObservationBuilder()
    manager = StateManager(GlobalState())
    loop, _, _ = _loop(
        manager,
        agents={
            "clinician": SilentAgent(),
            "patient": patient,
            "nurse": nurse,
            "relative": relative,
        },
        observation_builder=observation_builder,
    )

    result = loop.run_turn()

    assert result.active_agents == ["clinician"]
    assert observation_builder.calls[-1]["active_agents"] == ["clinician"]
    assert patient.generated_observations == []
    assert nurse.generated_observations == []
    assert relative.generated_observations == []


def test_turn_loop_does_not_import_transition_engines_or_directly_assign_state() -> None:
    source = inspect.getsource(turn_loop_module)

    assert "transition_engines" not in source
    assert ".runtime_state.messages =" not in source
    assert ".runtime_state.pending_questions =" not in source
    assert ".runtime_state.required_response_agents =" not in source
    assert ".runtime_state.pending_diagnostic_results =" not in source
    assert ".runtime_state.newly_available_results =" not in source
    assert ".runtime_state.current_turn_events =" not in source
    assert ".runtime_state.last_turn_events =" not in source
    assert ".known_facts.available_results =" not in source
    assert ".patient_state.vitals =" not in source
    assert ".patient_state.features =" not in source
    assert ".patient_state.status_flags =" not in source


def test_termination_by_max_turns_stops_before_agent_invocation() -> None:
    clinician = SilentAgent()
    manager = StateManager(GlobalState(runtime_state={"turn_index": 2, "max_turns": 2}))
    loop, physiology, _ = _loop(manager, agents={"clinician": clinician})

    result = loop.run_turn()

    assert result.completed is False
    assert result.terminated is True
    assert result.termination_reason == "max_turns"
    assert clinician.generated_observations == []
    assert physiology.calls == []


def test_termination_by_patient_death_stops_before_agent_invocation() -> None:
    clinician = SilentAgent()
    manager = StateManager(
        GlobalState(patient_state={"status_flags": {"is_alive": False}})
    )
    loop, physiology, _ = _loop(manager, agents={"clinician": clinician})

    result = loop.run_turn()

    assert result.completed is False
    assert result.terminated is True
    assert result.termination_reason == "patient_not_alive"
    assert clinician.generated_observations == []
    assert physiology.calls == []
