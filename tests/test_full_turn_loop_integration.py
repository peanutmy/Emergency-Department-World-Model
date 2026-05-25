from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ed_world_model.orchestration.runner as runner_module
from ed_world_model.actions.registry import ActionFamily, KindHint
from ed_world_model.adapters.noop_emotion import NoopEmotionEngine
from ed_world_model.agents.clinician import ClinicianAgent, parse_clinician_response
from ed_world_model.agents.llm_client import FakeLLMClient
from ed_world_model.agents.nurse import NurseAgent, parse_nurse_response
from ed_world_model.agents.patient import PatientAgent, parse_patient_response
from ed_world_model.agents.relative import RelativeAgent, parse_relative_response
from ed_world_model.agents.schemas import AgentProposal
from ed_world_model.orchestration.runner import (
    IntegrationRunner,
    RecordingStubPhysiologyAdapter,
    ScriptedLLMCallable,
    scripted_agents,
)
from ed_world_model.state.global_state import Event, GlobalState


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "scenario_loader_minimal.json"


class RecordingNoopEmotionEngine(NoopEmotionEngine):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        return super().predict(
            current_patient_emotion,
            conversation_input=conversation_input,
            recent_messages=recent_messages,
            patient_profile_context=patient_profile_context,
        )


class FailingAgent:
    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        raise AssertionError(f"agent should not be invoked with {observation!r}")


class ObservationAssertingAgent:
    def __init__(self, proposal: AgentProposal | None = None) -> None:
        self.proposal = proposal or AgentProposal()
        self.generated_observations: list[dict[str, Any]] = []

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(deepcopy(observation))
        assert "truth_state" not in observation
        assert "runtime_state" not in observation
        assert "current_turn_events" not in observation
        assert "test_bank" not in str(observation)
        return self.proposal


class LeakageCheckingClinician:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.generated_observations: list[dict[str, Any]] = []

    def generate(self, observation: dict[str, Any]) -> AgentProposal:
        self.generated_observations.append(deepcopy(observation))
        assert self.secret not in str(observation)
        assert observation["available_diagnostic_tests"] == ["ECG"]
        assert "truth_state" not in observation
        assert "test_bank" not in str(observation)
        return AgentProposal(
            action={"type": "diagnostic_order", "test_name": "ECG"}
        )


def _state_with_ecg(
    *,
    result: str = "atrial fibrillation",
    turnaround_turns: int = 1,
    runtime_state: dict[str, Any] | None = None,
    patient_state: dict[str, Any] | None = None,
) -> GlobalState:
    return GlobalState(
        truth_state={
            "test_bank": [
                {
                    "name": "ECG",
                    "result": result,
                    "turnaround_turns": turnaround_turns,
                }
            ]
        },
        runtime_state=runtime_state or {},
        patient_state=patient_state or {},
    )


def _oxygen_order() -> dict[str, Any]:
    return {
        "type": "medical_treatment_order",
        "family": ActionFamily.RESPIRATORY_SUPPORT,
        "kind_hint": KindHint.OXYGEN_SUPPORT,
        "params": {"oxygen_device": "NRB", "FiO2": 1.0},
    }


def _verbal(role: str, content: str, target: str = "clinician") -> dict[str, Any]:
    return {
        "verbal_action": {
            "speaker": role,
            "target": target,
            "content": content,
            "requires_response": False,
        }
    }


def _validation_drops(turn) -> list[dict[str, Any]]:
    return [event for event in turn.events if event["type"] == "validation_drop"]


def test_runner_accepts_scenario_json_path_and_runs_silent_turn() -> None:
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(
        FIXTURE_PATH,
        max_turns=3,
        physiology_adapter=physiology,
    )

    trajectory = runner.run(1)

    assert runner.state.truth_state.test_bank[0].name == "Initial ECG"
    assert trajectory[0].completed is True
    assert trajectory[0].active_agents == ["clinician"]
    assert trajectory[0].physiology_action_kind_hint == "no_action"
    assert physiology.calls[-1]["action"] == {
        "raw_text": None,
        "kind_hint": "no_action",
        "params": {"elapsed_min": 1},
    }
    assert runner.state.runtime_state.turn_index == 1


def test_full_runner_can_run_one_turn_with_initialized_state() -> None:
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(GlobalState(), physiology_adapter=physiology)

    turn = runner.run_turn()

    assert turn.completed is True
    assert turn.active_agents == ["clinician"]
    assert turn.turn_index_before == 0
    assert turn.turn_index_after == 1
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"


def test_full_runner_can_run_a_diagnostic_order_turn() -> None:
    physiology = RecordingStubPhysiologyAdapter()
    agents = scripted_agents(
        clinician=[{"action": {"type": "diagnostic_order", "test_name": "ECG"}}],
        nurse=["this should not run"],
    )
    runner = IntegrationRunner(
        _state_with_ecg(),
        agents=agents,
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    assert [
        pending.model_dump()
        for pending in runner.state.runtime_state.pending_diagnostic_results
    ] == [{"test_name": "ECG", "ordered_at_turn": 0, "ready_at_turn": 1}]
    assert runner.state.known_facts.available_results == []
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"
    assert agents["nurse"].generated_observations == []
    assert not any(
        event["type"] == "nurse_bedside_verbal_slot" for event in turn.events
    )


def test_duplicate_pending_diagnostic_order_is_dropped_and_uses_no_action() -> None:
    physiology = RecordingStubPhysiologyAdapter()
    agents = scripted_agents(
        clinician=[{"action": {"type": "diagnostic_order", "test_name": "ECG"}}],
    )
    runner = IntegrationRunner(
        _state_with_ecg(
            runtime_state={
                "pending_diagnostic_results": [
                    {
                        "test_name": "ECG",
                        "ordered_at_turn": 0,
                        "ready_at_turn": 5,
                    }
                ]
            }
        ),
        agents=agents,
        physiology_adapter=physiology,
    )
    pending_before = [
        pending.model_dump()
        for pending in runner.state.runtime_state.pending_diagnostic_results
    ]

    turn = runner.run_turn()

    assert turn.validation_results[0]["ok"] is False
    assert any(
        "already pending" in error for error in turn.validation_results[0]["errors"]
    )
    assert [
        pending.model_dump()
        for pending in runner.state.runtime_state.pending_diagnostic_results
    ] == pending_before
    assert not any(
        event["type"] == "diagnostic_order_created" for event in turn.events
    )
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"


def test_diagnostic_result_releases_before_action_selection() -> None:
    agents = scripted_agents(
        clinician=[
            {"action": {"type": "diagnostic_order", "test_name": "ECG"}},
            None,
        ],
        nurse=[None],
    )
    runner = IntegrationRunner(_state_with_ecg(), agents=agents)

    trajectory = runner.run(2)

    release_turn = trajectory[1]
    assert release_turn.released_diagnostics == [
        {"name": "ECG", "result": "atrial fibrillation"}
    ]
    assert release_turn.active_agents == ["clinician", "nurse"]
    assert agents["clinician"].generated_observations[1][
        "newly_available_results"
    ] == [{"name": "ECG", "result": "atrial fibrillation"}]
    assert runner.state.runtime_state.newly_available_results == []
    assert runner.state.known_facts.available_results[0].model_dump() == {
        "name": "ECG",
        "result": "atrial fibrillation",
    }


def test_full_runner_can_run_medical_treatment_order_turn() -> None:
    clinician_llm = ScriptedLLMCallable(
        [{"verbal_action": None, "action": _oxygen_order()}]
    )
    physiology = RecordingStubPhysiologyAdapter(
        output={"features": {"oxygen_device": "NRB"}}
    )
    runner = IntegrationRunner(
        GlobalState(),
        agents={"clinician": ClinicianAgent(clinician_llm)},
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    assert turn.validation_results[0]["ok"] is True
    assert turn.validation_results[0]["action_type"] == "medical_treatment_order"
    assert any(event["type"] == "nurse_shadow_execution" for event in turn.events)
    assert physiology.calls[-1]["action"]["kind_hint"] == "oxygen_support"
    assert physiology.calls[-1]["action"]["raw_text"] is None
    assert runner.state.patient_state.features.oxygen_device == "NRB"


def test_llm_client_output_still_flows_through_clinician_parser() -> None:
    llm_client = FakeLLMClient(
        [{"verbal_action": None, "action": _oxygen_order()}]
    )
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(
        GlobalState(),
        agents={"clinician": ClinicianAgent(llm_client)},
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    assert llm_client.prompts
    assert turn.validation_results[0]["ok"] is True
    assert turn.validation_results[0]["normalized_action"]["raw_text"] is None
    assert physiology.calls[-1]["action"]["kind_hint"] == "oxygen_support"
    assert physiology.calls[-1]["action"]["raw_text"] is None


def test_patient_nurse_relative_verbal_only_agents_can_speak() -> None:
    direct_patient_proposal = PatientAgent(
        ScriptedLLMCallable([_verbal("patient", "It hurts.")])
    ).generate({"recent_messages": []})
    assert isinstance(direct_patient_proposal, AgentProposal)
    assert direct_patient_proposal.action is None

    agents = {
        "patient": PatientAgent(
            ScriptedLLMCallable([_verbal("patient", "My chest feels tight.")])
        ),
        "nurse": NurseAgent(
            ScriptedLLMCallable([_verbal("nurse", "I can help.", "patient")])
        ),
        "relative": RelativeAgent(
            ScriptedLLMCallable([_verbal("relative", "What is happening?")])
        ),
    }
    runner = IntegrationRunner(
        GlobalState(
            runtime_state={
                "required_response_agents": ["patient", "nurse", "relative"]
            }
        ),
        agents=agents,
    )

    runner.run_turn()

    assert [
        message.speaker for message in runner.state.runtime_state.messages
    ] == ["nurse", "patient", "relative"]
    assert not any(
        event.type == "validation_drop"
        for event in runner.state.runtime_state.last_turn_events
    )


def test_nurse_bedside_verbal_slot_triggers_only_after_treatment() -> None:
    agents = {
        "clinician": ClinicianAgent(
            ScriptedLLMCallable([{"verbal_action": None, "action": _oxygen_order()}])
        ),
        "nurse": NurseAgent(
            ScriptedLLMCallable(
                [_verbal("nurse", "I am placing the oxygen mask now.", "patient")]
            )
        ),
    }
    runner = IntegrationRunner(GlobalState(), agents=agents)

    turn = runner.run_turn()

    bedside_events = [
        event for event in turn.events if event["type"] == "nurse_bedside_verbal_slot"
    ]
    assert bedside_events[-1]["payload"]["triggered"] is True
    assert bedside_events[-1]["payload"]["spoke"] is True
    assert runner.state.runtime_state.messages[-1].speaker == "nurse"
    assert runner.state.runtime_state.messages[-1].recipient == "patient"


def test_invalid_clinician_action_is_dropped_and_uses_no_action() -> None:
    clinician_llm = ScriptedLLMCallable(
        [
            {
                "verbal_action": None,
                "action": {
                    "type": "medical_treatment_order",
                    "family": ActionFamily.RESPIRATORY_SUPPORT,
                    "kind_hint": "not_a_kind_hint",
                    "params": {},
                },
            }
        ]
    )
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(
        GlobalState(),
        agents={"clinician": ClinicianAgent(clinician_llm)},
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    assert turn.validation_results[0]["ok"] is False
    assert any(event["type"] == "validation_drop" for event in turn.events)
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"
    assert runner.state.runtime_state.pending_diagnostic_results == []


def test_clinician_parser_error_through_runner_is_dropped_and_logged() -> None:
    clinician_llm = ScriptedLLMCallable(["not valid json"], role="clinician")
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(
        GlobalState(),
        agents={"clinician": ClinicianAgent(clinician_llm)},
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    drops = _validation_drops(turn)
    assert len(drops) == 1
    assert drops[0]["payload"]["agent"] == "clinician"
    assert drops[0]["payload"]["item"] == "agent_proposal"
    assert drops[0]["payload"]["phase"] == "primary_agent_generation"
    assert drops[0]["payload"]["error_type"] == "ClinicianParserError"
    assert "strict JSON" in drops[0]["payload"]["error_message"]
    assert turn.validation_results[0]["ok"] is True
    assert turn.validation_results[0]["action_type"] is None
    assert physiology.calls[-1]["action"]["kind_hint"] == "no_action"
    assert runner.state.runtime_state.messages == []


def test_verbal_only_parser_errors_through_runner_are_silent() -> None:
    agents = {
        "patient": PatientAgent(
            ScriptedLLMCallable(
                [_verbal("nurse", "Wrong speaker.")],
                role="patient",
            )
        ),
        "nurse": NurseAgent(
            ScriptedLLMCallable(
                [{"verbal_action": None, "action": None}],
                role="nurse",
            )
        ),
        "relative": RelativeAgent(
            ScriptedLLMCallable(
                [
                    {
                        "verbal_action": _verbal("relative", "Hi")[
                            "verbal_action"
                        ],
                        "action": None,
                    }
                ],
                role="relative",
            )
        ),
    }
    runner = IntegrationRunner(
        GlobalState(
            runtime_state={
                "required_response_agents": ["patient", "nurse", "relative"]
            }
        ),
        agents=agents,
    )

    turn = runner.run_turn()

    drops = _validation_drops(turn)
    assert [drop["payload"]["agent"] for drop in drops] == [
        "nurse",
        "patient",
        "relative",
    ]
    assert {drop["payload"]["error_type"] for drop in drops} == {
        "NurseParserError",
        "PatientParserError",
        "RelativeParserError",
    }
    assert all(
        drop["payload"]["phase"] == "primary_agent_generation"
        for drop in drops
    )
    assert runner.state.runtime_state.messages == []


def test_scripted_llm_callable_clinician_default_is_parseable() -> None:
    llm = ScriptedLLMCallable([], role="clinician")

    proposal = parse_clinician_response(llm("prompt"))

    assert proposal.verbal_action is None
    assert proposal.action is None


def test_scripted_llm_callable_verbal_only_defaults_are_parseable() -> None:
    defaults = {
        "patient": (ScriptedLLMCallable([], role="patient"), parse_patient_response),
        "nurse": (ScriptedLLMCallable([], role="nurse"), parse_nurse_response),
        "relative": (ScriptedLLMCallable([], role="relative"), parse_relative_response),
    }

    for llm, parser in defaults.values():
        proposal = parser(llm("prompt"))
        assert proposal.verbal_action is None
        assert proposal.action is None


def test_multiturn_verbal_only_agent_exhaustion_uses_parseable_silence() -> None:
    patient_llm = ScriptedLLMCallable(
        [_verbal("patient", "My breathing feels tight.")],
        role="patient",
    )
    runner = IntegrationRunner(
        GlobalState(runtime_state={"required_response_agents": ["patient"]}),
        agents={"patient": PatientAgent(patient_llm)},
    )

    trajectory = runner.run(2)

    assert len(trajectory) == 2
    assert len(patient_llm.prompts) == 2
    assert [message.speaker for message in runner.state.runtime_state.messages] == [
        "patient"
    ]
    assert not _validation_drops(trajectory[1])


def test_nurse_bedside_parser_error_is_logged_and_treatment_proceeds() -> None:
    physiology = RecordingStubPhysiologyAdapter()
    runner = IntegrationRunner(
        GlobalState(),
        agents={
            "clinician": ClinicianAgent(
                ScriptedLLMCallable(
                    [{"verbal_action": None, "action": _oxygen_order()}],
                    role="clinician",
                )
            ),
            "nurse": NurseAgent(
                ScriptedLLMCallable(
                    [{"verbal_action": None, "action": None}],
                    role="nurse",
                )
            ),
        },
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    drops = _validation_drops(turn)
    assert len(drops) == 1
    assert drops[0]["payload"]["agent"] == "nurse"
    assert drops[0]["payload"]["phase"] == "nurse_bedside_verbal_slot"
    assert drops[0]["payload"]["error_type"] == "NurseParserError"
    bedside_events = [
        event for event in turn.events if event["type"] == "nurse_bedside_verbal_slot"
    ]
    assert bedside_events[-1]["payload"]["spoke"] is False
    assert physiology.calls[-1]["action"]["kind_hint"] == "oxygen_support"


def test_no_real_llm_or_api_calls_are_required() -> None:
    clinician_llm = ScriptedLLMCallable([{"verbal_action": None, "action": None}])
    runner = IntegrationRunner(
        GlobalState(),
        agents={"clinician": ClinicianAgent(clinician_llm)},
    )

    runner.run_turn()

    assert len(clinician_llm.prompts) == 1
    source = inspect.getsource(runner_module)
    assert "openai" not in source.lower()
    assert "anthropic" not in source.lower()
    assert "claude" not in source.lower()


def test_no_hidden_diagnostic_result_leakage_before_release() -> None:
    secret = "SECRET_UNRELEASED_ECG_RESULT"
    physiology = RecordingStubPhysiologyAdapter()
    clinician = LeakageCheckingClinician(secret)
    runner = IntegrationRunner(
        _state_with_ecg(result=secret, turnaround_turns=2),
        agents={"clinician": clinician},
        physiology_adapter=physiology,
    )

    turn = runner.run_turn()

    assert runner.state.known_facts.available_results == []
    assert secret not in str(turn.as_dict())
    assert secret not in str(physiology.calls)
    assert physiology.calls[-1]["known_results"] == []


def test_agents_receive_partial_observations_not_full_global_state() -> None:
    agents = {
        "clinician": ObservationAssertingAgent(),
        "nurse": ObservationAssertingAgent(),
        "patient": ObservationAssertingAgent(),
        "relative": ObservationAssertingAgent(),
    }
    runner = IntegrationRunner(
        _state_with_ecg(
            runtime_state={
                "required_response_agents": ["nurse", "patient", "relative"]
            }
        ),
        agents=agents,
    )

    turn = runner.run_turn()

    assert turn.active_agents == ["clinician", "nurse", "patient", "relative"]
    assert all(agent.generated_observations for agent in agents.values())


def test_noop_emotion_engine_behavior_with_and_without_messages() -> None:
    emotion = RecordingNoopEmotionEngine()
    runner = IntegrationRunner(
        GlobalState(),
        agents=scripted_agents(
            clinician=[
                {
                    "verbal_action": {
                        "speaker": "clinician",
                        "recipient": "patient",
                        "content": "We are helping you.",
                    }
                }
            ]
        ),
        emotion_engine=emotion,
    )
    before = runner.state.psych_state.patient_emotion.model_dump()

    runner.run_turn()

    assert len(emotion.calls) == 1
    assert runner.state.psych_state.patient_emotion.model_dump() == before

    silent_emotion = RecordingNoopEmotionEngine()
    silent_runner = IntegrationRunner(GlobalState(), emotion_engine=silent_emotion)
    silent_runner.run_turn()

    assert silent_emotion.calls == []


def test_termination_prevents_agent_and_physiology_invocation() -> None:
    max_turn_physiology = RecordingStubPhysiologyAdapter()
    max_turn_runner = IntegrationRunner(
        GlobalState(runtime_state={"turn_index": 2, "max_turns": 2}),
        agents={"clinician": FailingAgent()},
        physiology_adapter=max_turn_physiology,
    )

    max_turn = max_turn_runner.run_turn()

    assert max_turn.terminated is True
    assert max_turn.termination_reason == "max_turns"
    assert max_turn_physiology.calls == []

    death_physiology = RecordingStubPhysiologyAdapter()
    death_runner = IntegrationRunner(
        GlobalState(patient_state={"status_flags": {"is_alive": False}}),
        agents={"clinician": FailingAgent()},
        physiology_adapter=death_physiology,
    )

    death_turn = death_runner.run_turn()

    assert death_turn.terminated is True
    assert death_turn.termination_reason == "patient_not_alive"
    assert death_physiology.calls == []


def test_last_turn_events_are_promoted_and_current_events_are_not_observed() -> None:
    agents = scripted_agents(
        clinician=[
            {"action": _oxygen_order()},
            None,
        ]
    )
    runner = IntegrationRunner(GlobalState(), agents=agents)

    first_turn = runner.run_turn()

    assert runner.state.runtime_state.last_turn_events == [
        Event.model_validate(event) for event in first_turn.events
    ]

    runner.run_turn()

    second_observation = agents["clinician"].generated_observations[1]
    assert any(
        event["type"] == "nurse_shadow_execution"
        for event in second_observation["last_turn_events"]
    )
    assert "current_turn_events" not in second_observation


def test_trajectory_turn_contains_debug_information() -> None:
    agents = scripted_agents(
        clinician=[
            {
                "verbal_action": {
                    "speaker": "clinician",
                    "recipient": "patient",
                    "content": "I am starting oxygen.",
                },
                "action": _oxygen_order(),
            }
        ],
        nurse=["Mask is going on now."],
    )
    runner = IntegrationRunner(GlobalState(), agents=agents)

    turn = runner.run_turn()
    dumped = turn.as_dict()

    assert dumped["active_agents"] == ["clinician"]
    assert [message["speaker"] for message in dumped["committed_messages"]] == [
        "clinician",
        "nurse",
    ]
    assert dumped["validation_results"][0]["ok"] is True
    assert dumped["physiology_action_kind_hint"] == "oxygen_support"
    assert {event["type"] for event in dumped["events"]} >= {
        "nurse_shadow_execution",
        "nurse_bedside_verbal_slot",
        "physiology_engine_call",
    }
