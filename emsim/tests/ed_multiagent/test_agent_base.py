from __future__ import annotations

from pathlib import Path
import sys

import pytest

EMSIM_ROOT = Path(__file__).resolve().parents[2]
if str(EMSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(EMSIM_ROOT))

from ed_multiagent.actions import AgentTurnOutput
from ed_multiagent.agents import (
    BaseAgent,
    ClinicianAgent,
    MockAgent,
    NurseAgent,
    PatientAgent,
    RelativeAgent,
)
from ed_multiagent.observation import PatientObservation, SpeechCapacity, SubjectiveState


def _subjective_state(
    speech_capacity: SpeechCapacity = SpeechCapacity.NORMAL,
) -> SubjectiveState:
    return SubjectiveState(
        dyspnea_severity=0.2,
        confusion_level=0.0,
        palpitation=False,
        dizziness=0.1,
        pain_distress=0.3,
        speech_capacity=speech_capacity,
        visible_distress=0.25,
    )


def test_base_agent_enforces_static_dynamic_prompt_split():
    agent = BaseAgent(role="test")
    first_observation = {"role": "test", "current_vitals": {"HR": 111}}
    second_observation = {"role": "test", "current_vitals": {"HR": 222}}

    static_prompt = agent.build_static_prompt()
    first_dynamic = agent.build_dynamic_prompt(first_observation)
    second_dynamic = agent.build_dynamic_prompt(second_observation)

    assert static_prompt == agent.build_static_prompt()
    assert "111" not in static_prompt
    assert "222" not in static_prompt
    assert "111" in first_dynamic
    assert "222" in second_dynamic
    assert first_dynamic != second_dynamic


def test_base_agent_act_returns_default_structured_no_llm_output():
    output = BaseAgent(role="test").act({"role": "test"})

    assert isinstance(output, AgentTurnOutput)
    assert output.metadata["role"] == "test"
    assert output.metadata["no_llm"] is True


def test_base_agent_validate_output_rejects_invalid_output_type():
    agent = BaseAgent(role="test")

    with pytest.raises(TypeError, match="AgentTurnOutput or mapping"):
        agent.validate_output(["not", "a", "mapping"])


@pytest.mark.parametrize("role", ["", "   "])
def test_base_agent_normalize_role_rejects_empty_role(role: str):
    with pytest.raises(ValueError, match="agent role must be non-empty"):
        BaseAgent(role=role)


def test_build_dynamic_prompt_rejects_mismatched_observation_role():
    agent = NurseAgent()

    with pytest.raises(
        ValueError,
        match="observation role 'patient' does not match agent role 'nurse'",
    ):
        agent.build_dynamic_prompt({"role": "patient"})


def test_static_prompt_contains_contract_not_observation_data():
    agent = NurseAgent()
    observation = {
        "role": "nurse",
        "current_vitals": {"HR": 145},
        "recent_dialogue": [{"content": "patient says chest pressure"}],
    }

    static_prompt = agent.build_static_prompt()
    dynamic_prompt = agent.build_dynamic_prompt(observation)

    assert "Allowed action schema" in static_prompt
    assert "Safety / information-boundary rules" in static_prompt
    assert "145" not in static_prompt
    assert "chest pressure" not in static_prompt
    assert "145" in dynamic_prompt
    assert "chest pressure" in dynamic_prompt


def test_dynamic_prompt_includes_current_observation_sections():
    agent = NurseAgent()
    observation = {
        "role": "nurse",
        "current_vitals": {"O2Sat": 88},
        "recent_dialogue": [{"speaker": "clinician", "content": "Vitals?"}],
        "response_opportunities": [{"question_text": "Can you get vitals?"}],
        "active_task": {"task_type": "draw_lab"},
    }

    prompt = agent.build_dynamic_prompt(observation)

    assert "Current role-specific observation" in prompt
    assert "Recent dialogue" in prompt
    assert "Response opportunity" in prompt
    assert "Task / workflow state" in prompt
    assert "O2Sat" in prompt
    assert "draw_lab" in prompt


def test_subjective_state_is_not_rendered_twice_in_dynamic_prompt():
    observation = PatientObservation(
        role="patient",
        subjective_state=_subjective_state(SpeechCapacity.SHORT_PHRASES),
    )

    prompt = PatientAgent().build_dynamic_prompt(observation)

    assert prompt.count('"subjective_state"') <= 1
    assert "dyspnea_severity" in prompt


def test_dynamic_sections_render_only_for_non_empty_values():
    agent = NurseAgent()
    empty_prompt = agent.build_dynamic_prompt(
        {
            "role": "nurse",
            "pending_orders": [],
            "active_task": None,
            "task_queue_summary": {},
            "queued_tasks": [],
            "completed_tasks": [],
            "pending_events": [],
            "emotion_state": None,
            "emotional_state": {},
            "emotional_or_social_context": "",
        }
    )

    assert "Task / workflow state" not in empty_prompt
    assert "Emotion state" not in empty_prompt

    populated_prompt = agent.build_dynamic_prompt(
        {
            "role": "nurse",
            "active_task": {"task_type": "draw_lab"},
            "emotional_state": {"urgency": "high"},
        }
    )

    assert "Task / workflow state" in populated_prompt
    assert "Emotion state" in populated_prompt


def test_dynamic_prompt_sanitizes_forbidden_hidden_fields():
    agent = PatientAgent()
    observation = {
        "role": "patient",
        "patient_private_memory": {
            "complaint": "safe_complaint",
            "hidden_state": "alpha_token",
            "pathology": "beta_token",
        },
    }

    prompt = agent.build_dynamic_prompt(observation)

    assert "safe_complaint" in prompt
    assert "hidden_state" not in prompt
    assert "pathology" not in prompt
    assert "alpha_token" not in prompt
    assert "beta_token" not in prompt


def test_all_m5_agents_implement_prompt_methods():
    agents = [
        MockAgent(role="mock"),
        NurseAgent(),
        PatientAgent(),
        RelativeAgent(),
        ClinicianAgent(),
    ]

    for agent in agents:
        assert isinstance(agent.build_static_prompt(), str)
        assert isinstance(agent.build_dynamic_prompt({"role": agent.role}), str)


def test_mock_agent_returns_structured_agent_turn_output():
    configured = AgentTurnOutput(
        verbal_action={
            "speaker": "nurse",
            "target": "clinician",
            "content": "Vitals are available.",
        },
        information_action={"action_type": "report"},
    )
    agent = MockAgent(role="nurse", output=configured)

    output = agent.act({"role": "nurse"})

    assert isinstance(output, AgentTurnOutput)
    assert output.verbal_action["content"] == "Vitals are available."
    assert output.information_action == {"action_type": "report"}


def test_mock_agent_output_is_copied_between_turns():
    agent = MockAgent(
        role="nurse",
        output={"information_action": {"action_type": "report"}},
    )

    first = agent.act({"role": "nurse"})
    first.information_action["mutated"] = True
    second = agent.act({"role": "nurse"})

    assert second.information_action == {"action_type": "report"}


def test_agent_turn_output_rejects_list_actions():
    with pytest.raises(TypeError, match="information_action must be a mapping"):
        AgentTurnOutput(information_action=[{"action_type": "report"}])

    with pytest.raises(TypeError, match="physical_action must be a mapping"):
        AgentTurnOutput(physical_action=[{"action_type": "execute"}])


def test_nurse_static_prompt_contains_vitals_and_hidden_state_boundary():
    prompt = NurseAgent().build_static_prompt().lower()

    assert "observable vitals" in prompt
    assert "validated tasks only" in prompt
    assert "ask for clarification" in prompt
    assert "hidden diagnosis" in prompt
    assert "hidden physiology" in prompt


def test_patient_prompt_does_not_expose_hidden_state_and_mentions_speech_capacity():
    prompt = PatientAgent().build_static_prompt().lower()

    assert "symptoms, feelings, and private memory only" in prompt
    assert "exact vitals unless staff told you" in prompt
    assert "hidden diagnosis" in prompt
    assert "hidden physiology" in prompt
    assert "speech_capacity is unable" in prompt


def test_patient_agent_respects_unable_speech_capacity():
    observation = PatientObservation(
        subjective_state=_subjective_state(SpeechCapacity.UNABLE),
        response_opportunities=[
            {"question_text": "Can you tell me your name?", "addressed_to": "patient"}
        ],
    )

    output = PatientAgent().act(observation)

    assert isinstance(output, AgentTurnOutput)
    assert output.verbal_action is None
    assert output.meta_action == {
        "type": "unable_to_answer",
        "response_mode": "unable",
        "reason": "speech_capacity_unable",
    }


def test_patient_agent_with_speech_capacity_not_unable_returns_default_output():
    observation = PatientObservation(
        role="patient",
        subjective_state=_subjective_state(SpeechCapacity.SHORT_PHRASES),
    )

    output = PatientAgent().act(observation)

    assert isinstance(output, AgentTurnOutput)
    assert output.meta_action is None
    assert output.metadata == {"role": "patient", "no_llm": True}


def test_relative_prompt_does_not_claim_hidden_physiology_access():
    prompt = RelativeAgent().build_static_prompt().lower()

    assert "relative private memory and visible patient state" in prompt
    assert "do not know hidden diagnosis or physiology" in prompt
    assert "can access hidden physiology" not in prompt


def test_clinician_prompt_limits_agent_to_discovered_clinical_info():
    prompt = ClinicianAgent().build_static_prompt().lower()

    assert "only know discovered clinical information" in prompt
    assert "cannot access hidden diagnosis or hidden physiology" in prompt
    assert "structured orders, questions, and information requests" in prompt


def test_relative_and_clinician_agents_return_structured_noop_outputs():
    relative_output = RelativeAgent().act({"role": "relative"})
    clinician_output = ClinicianAgent().act({"role": "clinician"})

    assert isinstance(relative_output, AgentTurnOutput)
    assert relative_output.metadata == {"role": "relative", "no_llm": True}
    assert isinstance(clinician_output, AgentTurnOutput)
    assert clinician_output.metadata == {"role": "clinician", "no_llm": True}


def test_agent_source_does_not_import_physiology_internals():
    agent_files = sorted((EMSIM_ROOT / "ed_multiagent/agents").glob("*.py"))
    assert agent_files

    for path in agent_files:
        source = path.read_text(encoding="utf-8")
        assert "rule_engine" not in source
        assert "HiddenState" not in source
        assert "drug_lib" not in source
        assert "pathology_lib" not in source
        assert "intervention_lib" not in source
        assert "apply_emsim_action" not in source
        assert "current_state" not in source
        assert "snapshot()" not in source
