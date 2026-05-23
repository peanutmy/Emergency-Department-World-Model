from ed_world_model.state.global_state import (
    GlobalState,
    TruthState,
    TestBankItem,
    PatientState,
    Vitals,
    Features,
    StatusFlags,
)
from ed_world_model.state.state_manager import StateManager
from ed_world_model.orchestration.orchestrator import Orchestrator
from ed_world_model.orchestration.observation_builder import ObservationBuilder
from ed_world_model.orchestration.turn_loop import TurnLoop
from ed_world_model.actions.registry import ActionRegistry
from ed_world_model.actions.validator import ActionValidator
from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.agents.stubs import ScriptedClinicianAgent, ScriptedNurseAgent, SilentAgent


class FakePhysiologyAdapter:
    def __init__(self):
        self.calls = []

    def predict(self, global_state, engine_facing_action):
        self.calls.append(engine_facing_action)

        # 最小 fake output：根据你 StateManager.apply_physiology_update 接口调整
        return {
            "vitals": {
                "HR": 110,
                "BP_sys": 150,
                "BP_dia": 95,
                "RR": 34,
                "O2Sat": 88,
                "T": 36.9,
            },
            "features": {
                "oxygen_device": engine_facing_action["params"].get("oxygen_device"),
                "FiO2": engine_facing_action["params"].get("FiO2"),
            },
        }


def make_state():
    return GlobalState(
        truth_state=TruthState(
            scenario_description="78-year-old woman with acute respiratory distress after surgery.",
            test_bank=[
                TestBankItem(
                    name="ECG",
                    result="ECG image link provided; no text interpretation stated.",
                    turnaround_turns=1,
                ),
                TestBankItem(
                    name="Chest X-ray",
                    result="CXR resource link provided; no text interpretation stated.",
                    turnaround_turns=2,
                ),
            ],
        ),
        patient_state=PatientState(
            vitals=Vitals(HR=110, BP_sys=150, BP_dia=95, RR=34, O2Sat=80, T=36.9),
            features=Features(
                rhythm="NSR",
                oxygen_device=None,
                FiO2=0.21,
                intubated=False,
                vent=False,
                PEEP_cmH2O=None,
                neuro_status="GCS 15",
                glucose_mmol_l=6.2,
            ),
            status_flags=StatusFlags(is_alive=True, can_speak=True),
        ),
    )


def main():
    state = make_state()
    manager = StateManager(state)
    registry = ActionRegistry()
    validator = ActionValidator(registry)
    physiology = FakePhysiologyAdapter()

    clinician = ScriptedClinicianAgent([
        # Turn 0: order ECG
        {
            "verbal_action": {
                "speaker": "clinician",
                "recipient": "patient",
                "content": "I want to check your heart rhythm.",
                "requires_response": False,
            },
            "action": {
                "type": "diagnostic_order",
                "test_name": "ECG",
            },
        },
        # Turn 1: treatment order
        {
            "verbal_action": {
                "speaker": "clinician",
                "recipient": "patient",
                "content": "We are going to put an oxygen mask on you.",
                "requires_response": False,
            },
            "action": {
                "type": "medical_treatment_order",
                "family": "respiratory_support",
                "kind_hint": "oxygen_support",
                "params": {
                    "oxygen_device": "NRB",
                    "FiO2": 0.8,
                    "PEEP_used": False,
                    "PEEP_cmH2O": None,
                },
            },
        },
        # Turn 2: no action
        {
            "verbal_action": None,
            "action": None,
        },
    ])

    nurse = ScriptedNurseAgent([
        {
            "verbal_action": {
                "speaker": "nurse",
                "recipient": "patient",
                "content": "I am placing this oxygen mask now. Try to take slow breaths.",
                "requires_response": False,
            }
        }
    ])

    agents = {
        "clinician": clinician,
        "nurse": nurse,
        "patient": SilentAgent(),
        "relative": SilentAgent(),
    }

    loop = TurnLoop(
        state_manager=manager,
        orchestrator=Orchestrator(),
        observation_builder=ObservationBuilder(),
        action_validator=validator,
        physiology_adapter=physiology,
        emotion_engine=NoopEmotionEngine(),
        agents=agents,
    )

    for i in range(3):
        print(f"\n===== TURN {manager.state.runtime_state.turn_index} =====")
        result = loop.run_turn()
        print("Turn result:", result)
        print("Messages:", [m.model_dump() for m in manager.state.runtime_state.messages])
        print("Known results:", [r.model_dump() for r in manager.state.known_facts.available_results])
        print("Pending diagnostics:", [p.model_dump() for p in manager.state.runtime_state.pending_diagnostic_results])
        print("Last turn events:", [e.model_dump() for e in manager.state.runtime_state.last_turn_events])
        print("Physiology calls:", physiology.calls)

    print("\nFinal state:")
    print(manager.state.model_dump())


if __name__ == "__main__":
    main()